// =============================================================================
// rshl_core.v — RSHL query engine (synthesizable Verilog)
//
// CSR row-major sparse matrix × sparse probe dot product; signed int16 sum per row;
// running top-K (replace minimum slot). Ternary encoding: 2'b00=0, 2'b01=+1,
// 2'b10=-1, 2'b11=0.
//
// BRAM model: registered read — address in cycle N, rdata valid on cycle N+1.
// =============================================================================
`timescale 1ns / 1ps

module rshl_core #(
    parameter integer DIM            = 2000,
    parameter integer NUM_ROWS       = 1024,
    parameter integer TOP_K          = 5,
    parameter integer PROBE_NNZ_MAX  = 64,
    parameter integer COL_MEM_DEPTH  = 65536,
    // Derived widths (override in testbench when DIM/NUM_ROWS/COL_MEM_DEPTH change)
    parameter integer COL_IDX_W      = 11,
    parameter integer ROW_ADDR_W     = 10,
    parameter integer COL_ADDR_W     = 16,
    parameter integer PROBE_ADDR_W   = 7
)(
    input  wire clk,
    input  wire rst_n,

    // Row range for this query (latched when start is asserted in S_IDLE).
    // Scans rows [row_scan_first, row_scan_first + row_scan_len) using global CSR indices.
    // Typical full-chip: first=0, len=NUM_ROWS. Multi-card: each card uses a disjoint slice.
    input  wire [ROW_ADDR_W-1:0] row_scan_first,
    input  wire [ROW_ADDR_W:0]   row_scan_len,

    output reg  [COL_ADDR_W-1:0] row_ptr_addr,
    input  wire [COL_ADDR_W-1:0] row_ptr_rdata,
    output reg  [COL_ADDR_W-1:0] col_idx_addr,
    input  wire [COL_IDX_W-1:0]  col_idx_rdata,
    output reg  [COL_ADDR_W-1:0] values_addr,
    input  wire [1:0]            values_rdata,

    input  wire [PROBE_ADDR_W-1:0] probe_addr,
    input  wire                    probe_wr_en,
    // Word 0: {26'b0, nnz[5:0]}; words 1..: {padding, val[1:0], col[COL_IDX_W-1:0]}
    input  wire [31:0]             probe_data,

    input  wire        start,
    output reg         busy,
    output reg         done,
    output reg         result_rdy,
    output reg signed [15:0] result_score,
    output reg [ROW_ADDR_W-1:0] result_row,
    output reg [7:0]   result_idx
);

    (* ram_style = "block" *)
    reg [31:0] probe_ram [0:PROBE_NNZ_MAX];

    always @(posedge clk) begin
        if (probe_wr_en)
            probe_ram[probe_addr] <= probe_data;
    end

    // FSM states
    localparam S_IDLE        = 5'd0;
    localparam S_BUSY_INIT   = 5'd1;
    localparam S_RP0_SET     = 5'd2;
    localparam S_RP0_CAP     = 5'd3;
    localparam S_RP1_SET     = 5'd4;
    localparam S_RP1_CAP     = 5'd5;
    localparam S_ROW_EMPTY   = 5'd6;
    localparam S_CS_SET      = 5'd7;
    localparam S_CS_CAP      = 5'd8;
    localparam S_PR_PROBE    = 5'd9;
    localparam S_CS_NEXT     = 5'd10;
    localparam S_ROW_TOPK    = 5'd11;
    localparam S_NEXT_ROW    = 5'd12;
    localparam S_OUT_STREAM  = 5'd13;
    localparam S_DONE_PULSE  = 5'd14;

    reg [4:0] state;

    reg [ROW_ADDR_W-1:0] cur_row;
    reg [ROW_ADDR_W-1:0] lat_first;
    reg [ROW_ADDR_W:0]   lat_len;
    reg [COL_ADDR_W-1:0] csr_k, csr_end, csr_begin;
    reg [COL_IDX_W-1:0]  csr_col_hold;
    reg [1:0]            csr_val_hold;
    reg [5:0]            probe_p;
    reg [5:0]            probe_nnz;
    reg signed [15:0]    acc;

    reg signed [15:0]    top_score [0:TOP_K-1];
    reg [ROW_ADDR_W-1:0] top_row   [0:TOP_K-1];

    reg [7:0]            out_slot;

    function signed [1:0] ternary_decode;
        input [1:0] t;
        begin
            case (t)
                2'b01:   ternary_decode = 2'sd1;
                2'b10:   ternary_decode = -2'sd1;
                default: ternary_decode = 2'sd0;
            endcase
        end
    endfunction

    wire [COL_IDX_W-1:0] p_col = probe_ram[probe_p + 1'b1][COL_IDX_W-1:0];
    wire [1:0]           p_val = probe_ram[probe_p + 1'b1][COL_IDX_W+1:COL_IDX_W];

    wire signed [1:0] rv = ternary_decode(csr_val_hold);
    wire signed [1:0] pv = ternary_decode(p_val);
    wire match = (csr_col_hold == p_col) && (rv != 2'sd0) && (pv != 2'sd0);

    // Exclusive end row index: lat_first + lat_len (e.g. first=4, len=4 -> rows 4..7)
    wire [ROW_ADDR_W+1:0] scan_end_ex;
    wire [ROW_ADDR_W+1:0] cur_inc;
    assign scan_end_ex = lat_first + lat_len;
    assign cur_inc     = {2'b0, cur_row} + (ROW_ADDR_W+2)'(1);

    integer ti;
    reg [7:0] min_slot;
    reg signed [15:0] min_score_w;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state        <= S_IDLE;
            busy         <= 1'b0;
            done         <= 1'b0;
            result_rdy   <= 1'b0;
            result_score <= 16'sd0;
            result_row   <= {ROW_ADDR_W{1'b0}};
            result_idx   <= 8'd0;
            row_ptr_addr <= {COL_ADDR_W{1'b0}};
            col_idx_addr <= {COL_ADDR_W{1'b0}};
            values_addr  <= {COL_ADDR_W{1'b0}};
            cur_row      <= {ROW_ADDR_W{1'b0}};
            lat_first    <= {ROW_ADDR_W{1'b0}};
            lat_len      <= NUM_ROWS;
            csr_k        <= {COL_ADDR_W{1'b0}};
            csr_begin    <= {COL_ADDR_W{1'b0}};
            csr_end      <= {COL_ADDR_W{1'b0}};
            csr_col_hold <= {COL_IDX_W{1'b0}};
            csr_val_hold <= 2'b00;
            probe_p      <= 6'd0;
            probe_nnz    <= 6'd0;
            acc          <= 16'sd0;
            out_slot     <= 8'd0;
            for (ti = 0; ti < TOP_K; ti = ti + 1) begin
                top_score[ti] <= 16'sh8000;
                top_row[ti]   <= {ROW_ADDR_W{1'b0}};
            end
        end else begin
            done       <= 1'b0;
            result_rdy <= 1'b0;

            case (state)
                S_IDLE: begin
                    busy <= 1'b0;
                    if (start) begin
                        busy      <= 1'b1;
                        probe_nnz <= probe_ram[0][5:0];
                        if (probe_ram[0][5:0] > PROBE_NNZ_MAX[5:0])
                            probe_nnz <= PROBE_NNZ_MAX[5:0];
                        lat_first <= row_scan_first;
                        lat_len   <= row_scan_len;
                        cur_row   <= row_scan_first;
                        for (ti = 0; ti < TOP_K; ti = ti + 1) begin
                            top_score[ti] <= 16'sh8000;
                            top_row[ti]   <= {ROW_ADDR_W{1'b0}};
                        end
                        state <= S_BUSY_INIT;
                    end
                end

                S_BUSY_INIT: begin
                    row_ptr_addr <= cur_row;
                    state        <= S_RP0_SET;
                end

                S_RP0_SET: begin
                    // row_ptr_addr already cur_row; wait one cycle for registered BRAM
                    state <= S_RP0_CAP;
                end

                S_RP0_CAP: begin
                    csr_begin    <= row_ptr_rdata;
                    row_ptr_addr <= cur_row + 1'b1;
                    state        <= S_RP1_SET;
                end

                S_RP1_SET: state <= S_RP1_CAP;

                S_RP1_CAP: begin
                    csr_end <= row_ptr_rdata;
                    acc     <= 16'sd0;
                    if (csr_begin == row_ptr_rdata)
                        state <= S_ROW_EMPTY;
                    else begin
                        csr_k <= csr_begin;
                        state <= S_CS_SET;
                    end
                end

                S_ROW_EMPTY: begin
                    // acc = 0
                    state <= S_ROW_TOPK;
                end

                S_CS_SET: begin
                    col_idx_addr <= csr_k;
                    values_addr  <= csr_k;
                    state        <= S_CS_CAP;
                end

                S_CS_CAP: begin
                    csr_col_hold <= col_idx_rdata;
                    csr_val_hold <= values_rdata;
                    probe_p      <= 6'd0;
                    state        <= S_PR_PROBE;
                end

                S_PR_PROBE: begin
                    if (probe_p < probe_nnz) begin
                        if (match)
                            acc <= acc + $signed({{14{rv[1]}}, rv}) * $signed({{14{pv[1]}}, pv});
                        probe_p <= probe_p + 1'b1;
                    end else
                        state <= S_CS_NEXT;
                end

                S_CS_NEXT: begin
                    if (csr_k + 1'b1 >= csr_end)
                        state <= S_ROW_TOPK;
                    else begin
                        csr_k <= csr_k + 1'b1;
                        state <= S_CS_SET;
                    end
                end

                S_ROW_TOPK: begin
                    min_score_w = top_score[0];
                    min_slot = 8'd0;
                    for (ti = 1; ti < TOP_K; ti = ti + 1) begin
                        if (top_score[ti] < min_score_w) begin
                            min_score_w = top_score[ti];
                            min_slot = ti[7:0];
                        end
                    end
                    if (acc > min_score_w) begin
                        top_score[min_slot] <= acc;
                        top_row[min_slot]   <= cur_row;
                    end
                    state <= S_NEXT_ROW;
                end

                S_NEXT_ROW: begin
                    if (cur_inc >= scan_end_ex) begin
                        out_slot <= 8'd0;
                        state    <= S_OUT_STREAM;
                    end else begin
                        cur_row <= cur_row + 1'b1;
                        row_ptr_addr <= cur_row + 1'b1;
                        state <= S_BUSY_INIT;
                    end
                end

                // Stream top-K (out_slot = 0 .. TOP_K-1) then finish
                S_OUT_STREAM: begin
                    result_rdy   <= 1'b1;
                    result_score <= top_score[out_slot];
                    result_row   <= top_row[out_slot];
                    result_idx   <= out_slot;
                    if (out_slot == TOP_K - 1)
                        state <= S_DONE_PULSE;
                    else
                        out_slot <= out_slot + 1'b1;
                end

                S_DONE_PULSE: begin
                    done  <= 1'b1;
                    busy  <= 1'b0;
                    state <= S_IDLE;
                end

                default: state <= S_IDLE;
            endcase
        end
    end

endmodule
