// =============================================================================
// tt_um_rshl.v — Tiny Tapeout wrapper for the KAI RSHL query core
//
// Pinout (fixed by Tiny Tapeout shuttle):
//   ui_in  [7:0]  — host input byte (data or address depending on reg_sel)
//   uo_out [7:0]  — host output byte (status or readback)
//   uio_in [7:0]  — bidirectional inputs  (probe data; see CARD_ID straps below)
//   uio_out[7:0]  — bidirectional outputs (result low byte)
//   uio_oe [7:0]  — output-enable for uio (driven when result valid)
//   ena           — design enable (tie to 1 in normal use)
//   clk           — system clock (50 MHz target)
//   rst_n         — active-low reset
//
// Multi-chip PCB (row sharding)
// =============================
// Set Verilog parameters: TT_NUM_CARDS (must divide TT_NUM_ROWS) and either
// TT_CARD_ID (synthesis default) or TT_STRAP_CARD_ID=1 to latch uio_in[1:0] on
// the first active clock after reset as this chip's CARD_ID (strap before
// releasing rst_n; keep straps static during normal I/O if strapping is used).
//
// Register interface (strobe + reg_sel protocol):
//   Write: assert ui_in[0]=1 (strobe), ui_in[3:1]=reg_sel, uio_in=data byte
//   Read:  assert ui_in[0]=0,          ui_in[3:1]=reg_sel  → uo_out/uio_out
//
// Register map:
//   sel 0  CMD      write: [0]=start [1]=sw_rst [3:2]=stage (unused) [6]=legacy
//   sel 1  STATUS   read:  [0]=busy [1]=done [2]=result_rdy
//   sel 2  P_ADDR   write: probe word address (7-bit)
//   sel 3  P_DATA_L write: probe data bits [7:0]   (latches low byte)
//   sel 4  P_DATA_H write: probe data bits [15:8]  (fires probe write on this)
//   sel 5  P_DATA_X write: probe data bits [31:16] (fires probe write on this)
//   sel 6  RESULT   read:  uo_out=score[7:0], uio_out=result_row[7:0]
//   sel 7  RESULT_H read:  uo_out=score[15:8], uio_out=result_idx[7:0]
// =============================================================================
`default_nettype none
`timescale 1ns / 1ps

module tt_um_rshl #(
    parameter integer TT_NUM_CARDS      = 1,
    parameter [1:0]   TT_CARD_ID        = 2'd0,
    parameter bit     TT_STRAP_CARD_ID  = 1'b0,
    parameter integer TT_NUM_ROWS       = 8,
    parameter integer TT_TOP_K          = 3,
    parameter integer TT_PROBE_NNZ_MAX  = 8,
    parameter integer TT_COL_MEM_DEPTH  = 64,
    parameter integer TT_COL_IDX_W      = 6,
    parameter integer TT_ROW_ADDR_W     = 3,
    parameter integer TT_COL_ADDR_W     = 6,
    parameter integer TT_PROBE_ADDR_W   = 4,
    parameter integer TT_DIM            = 64
)(
    input  wire [7:0] ui_in,
    output wire [7:0] uo_out,
    input  wire [7:0] uio_in,
    output wire [7:0] uio_out,
    output wire [7:0] uio_oe,
    input  wire       ena,
    input  wire       clk,
    input  wire       rst_n
);

    localparam integer TT_RPC = TT_NUM_ROWS / TT_NUM_CARDS;

    // ── Control decode ─────────────────────────────────────────────────────
    wire        strobe   = ui_in[0];
    wire [2:0]  reg_sel  = ui_in[3:1];
    wire        wr       = ui_in[0] && ena;
    wire        rd       = !ui_in[0] && ena;
    wire [7:0]  host_byte = uio_in;

    // ── CARD_ID: parameter or one-time strap sample ────────────────────────
    reg [1:0] card_id_lat;
    reg       card_id_init;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            card_id_init <= 1'b0;
            card_id_lat  <= TT_CARD_ID;
        end else if (ena && !card_id_init) begin
            if (TT_STRAP_CARD_ID)
                card_id_lat <= uio_in[1:0];
            else
                card_id_lat <= TT_CARD_ID;
            card_id_init <= 1'b1;
        end
    end

    wire [TT_ROW_ADDR_W-1:0] rpc_w = TT_RPC;
    wire [TT_ROW_ADDR_W-1:0] w_scan_first;
    wire [TT_ROW_ADDR_W:0]   w_scan_len;
    assign w_scan_first = rpc_w * card_id_lat;
    assign w_scan_len   = TT_RPC;

    // ── Register file ───────────────────────────────────────────────────────
    reg [7:0]  cmd_r;
    reg [6:0]  probe_addr_r;
    reg [31:0] probe_data_r;
    reg        probe_wr_fire;

    reg [7:0]  p_data_l, p_data_m;

    (* ram_style = "block" *)
    reg [TT_COL_ADDR_W-1:0] row_ptr_mem [0:TT_NUM_ROWS];
    (* ram_style = "block" *)
    reg [TT_COL_IDX_W-1:0]  col_idx_mem [0:TT_COL_MEM_DEPTH-1];
    (* ram_style = "block" *)
    reg [1:0]                values_mem  [0:TT_COL_MEM_DEPTH-1];

    reg [TT_COL_ADDR_W-1:0] row_ptr_rdata_r;
    reg [TT_COL_IDX_W-1:0]  col_idx_rdata_r;
    reg [1:0]                values_rdata_r;

    wire [TT_COL_ADDR_W-1:0] row_ptr_addr_w;
    wire [TT_COL_ADDR_W-1:0] col_idx_addr_w;
    wire [TT_COL_ADDR_W-1:0] values_addr_w;

    always @(posedge clk) begin
        row_ptr_rdata_r <= row_ptr_mem[row_ptr_addr_w];
        col_idx_rdata_r <= col_idx_mem[col_idx_addr_w];
        values_rdata_r  <= values_mem [values_addr_w];
    end

    wire        core_busy, core_done, core_result_rdy;
    wire signed [15:0]              core_result_score;
    wire [TT_ROW_ADDR_W-1:0]        core_result_row;
    wire [7:0]                       core_result_idx;

    wire sw_rst   = cmd_r[1];
    wire start_w  = cmd_r[0];

    rshl_core #(
        .DIM           (TT_DIM),
        .NUM_ROWS      (TT_NUM_ROWS),
        .TOP_K         (TT_TOP_K),
        .PROBE_NNZ_MAX (TT_PROBE_NNZ_MAX),
        .COL_MEM_DEPTH (TT_COL_MEM_DEPTH),
        .COL_IDX_W     (TT_COL_IDX_W),
        .ROW_ADDR_W    (TT_ROW_ADDR_W),
        .COL_ADDR_W    (TT_COL_ADDR_W),
        .PROBE_ADDR_W  (TT_PROBE_ADDR_W)
    ) u_core (
        .clk           (clk),
        .rst_n         (rst_n && !sw_rst),
        .row_scan_first(w_scan_first),
        .row_scan_len  (w_scan_len),
        .row_ptr_addr  (row_ptr_addr_w),
        .row_ptr_rdata (row_ptr_rdata_r),
        .col_idx_addr  (col_idx_addr_w),
        .col_idx_rdata (col_idx_rdata_r),
        .values_addr   (values_addr_w),
        .values_rdata  (values_rdata_r),
        .probe_addr    (probe_addr_r[TT_PROBE_ADDR_W-1:0]),
        .probe_wr_en   (probe_wr_fire),
        .probe_data    (probe_data_r),
        .start         (start_w),
        .busy          (core_busy),
        .done          (core_done),
        .result_rdy    (core_result_rdy),
        .result_score  (core_result_score),
        .result_row    (core_result_row),
        .result_idx    (core_result_idx)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            cmd_r        <= 8'h00;
            probe_addr_r <= 7'h00;
            probe_data_r <= 32'h0;
            probe_wr_fire<= 1'b0;
            p_data_l     <= 8'h00;
            p_data_m     <= 8'h00;
        end else begin
            probe_wr_fire <= 1'b0;
            if (wr) begin
                case (reg_sel)
                    3'd0: cmd_r        <= host_byte;
                    3'd2: probe_addr_r <= host_byte[6:0];
                    3'd3: p_data_l     <= host_byte;
                    3'd4: p_data_m     <= host_byte;
                    3'd5: begin
                              probe_data_r  <= {host_byte, 8'h0, p_data_m, p_data_l};
                              probe_wr_fire <= 1'b1;
                          end
                    default: ;
                endcase
            end
            if (cmd_r[0]) cmd_r[0] <= 1'b0;
        end
    end

    reg [7:0] out_main, out_bio;
    reg       out_oe;

    always @(*) begin
        out_main = 8'h00;
        out_bio  = 8'h00;
        out_oe   = 1'b0;
        case (reg_sel)
            3'd1: begin
                out_main = {5'b0, core_result_rdy, core_done, core_busy};
            end
            3'd6: begin
                out_main = core_result_score[7:0];
                out_bio  = {{(8-TT_ROW_ADDR_W){1'b0}}, core_result_row};
                out_oe   = 1'b1;
            end
            3'd7: begin
                out_main = core_result_score[15:8];
                out_bio  = core_result_idx;
                out_oe   = 1'b1;
            end
            default: out_main = 8'h00;
        endcase
    end

    assign uo_out  = out_main;
    assign uio_out = out_bio;
    assign uio_oe  = {8{out_oe}};

endmodule

`default_nettype wire
