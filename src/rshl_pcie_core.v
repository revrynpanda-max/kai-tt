// =============================================================================
// rshl_pcie_core.v — RSHL query engine with AXI-Lite host interface and
//                    multi-card coarse broadcast + row-sharded fine scan
//
// Architecture
// ============
//  ┌─────────────────────────────────────────────────────────────────┐
//  │  AXI-Lite slave (32-bit addr/data, host CPU interface)          │
//  │    - Write probe vector, trigger query, read TOP-K results      │
//  │    - Control register: stage_sel, legacy bits (see map)       │
//  └───────────────────────┬─────────────────────────────────────────┘
//                          │
//  ┌─────────────────────── ▼─────────────────────────────────────────┐
//  │  Hierarchy controller                                             │
//  │    COARSE stage: all cards run coarse in parallel (stub here)     │
//  │    FULL stage:   coarse → fine on *every* card (row shard)      │
//  │    FINE stage:   fine only, same row shard as this build         │
//  │    Fine scan:    rshl_core over rows                              │
//  │                  [CARD_ID*(NUM_ROWS/NUM_CARDS) .. )               │
//  └───────────────────────┬─────────────────────────────────────────┘
//                          │
//  ┌───────────────────────▼─────────────────────────────────────────┐
//  │  rshl_core (ternary dot-product + top-K within this card's rows) │
//  └─────────────────────────────────────────────────────────────────┘
//
// Host merges TOP-K streams from all cards (see rshl_merge_topk.v). Legacy
// winner_bucket_in / is_winner are not used for row-sharded FULL/FINE flows.
//
// AXI-Lite register map (byte offsets, 32-bit words)
// ───────────────────────────────────────────────────
//  0x00  CTRL    [0]   start       — pulse high for 1 cycle to launch query
//                [1]   sw_rst      — synchronous reset
//                [3:2] stage_sel   — 00=coarse, 01=fine, 10=full (coarse→fine)
//                [5:4] card_id     — legacy / bookkeeping (row shard uses param CARD_ID)
//                [6]   is_winner   — legacy (unused for row-sharded fine)
//                [7]   coarse_bcast— legacy (unused)
//  0x04  STATUS  [0]   busy
//                [1]   done
//                [2]   result_rdy
//  0x08  PROBE_ADDR    probe write address (7-bit)
//  0x0C  PROBE_DATA    probe write data (32-bit), auto-fires probe_wr_en
//  0x10  RESULT_SCORE  signed 16-bit, zero-extended to 32-bit
//  0x14  RESULT_ROW    row index (ROW_ADDR_W bits)
//  0x18  RESULT_IDX    top-K slot index (8-bit)
//  0x1C  COARSE_SCORE  top coarse score from this card (to host for arbitration)
//  0x20  COARSE_BUCKET top coarse bucket index from this card
//  0x24  WINNER_BUCKET legacy (optional future bucket mask); row-shard uses param CARD_ID
//  0x28  BOOT_CTRL     [0]=BOOT_EN (rw), [1]=BOOT_DONE (ro), [2]=BOOT_ERROR (ro)
// =============================================================================
`timescale 1ns / 1ps
`default_nettype none

module rshl_pcie_core #(
    // ── rshl_core parameters ──────────────────────────────────────────────
    parameter integer DIM           = 2000,
    parameter integer NUM_ROWS      = 1024,
    parameter integer TOP_K         = 5,
    parameter integer PROBE_NNZ_MAX = 64,
    parameter integer COL_MEM_DEPTH = 65536,
    parameter integer COL_IDX_W     = 11,
    parameter integer ROW_ADDR_W    = 10,
    parameter integer COL_ADDR_W    = 16,
    parameter integer PROBE_ADDR_W  = 7,
    // ── coarse hierarchy ─────────────────────────────────────────────────
    parameter integer COARSE_ROWS   = 256,   // centroid vectors per card
    parameter integer COARSE_DIM    = 256,   // coarse vector dimension
    parameter integer TOP_COARSE    = 10,    // top-K coarse buckets to report
    // ── PCIe / multi-card row sharding ───────────────────────────────────
    // NUM_ROWS must be divisible by NUM_CARDS. Each instance is synthesized
    // with CARD_ID in 0 .. NUM_CARDS-1; the host broadcasts probe/start to all.
    parameter integer NUM_CARDS     = 1,
    parameter integer CARD_ID       = 0,
    // ── Boot loader / seed import (kept internal for pin compatibility) ─────
    parameter integer BOOT_TIMEOUT_CYC = 500000
)(
    // ── Clock / reset ────────────────────────────────────────────────────
    input  wire        clk,
    input  wire        rst_n,

    // ── AXI-Lite slave ───────────────────────────────────────────────────
    input  wire [7:0]  axil_awaddr,
    input  wire        axil_awvalid,
    output reg         axil_awready,
    input  wire [31:0] axil_wdata,
    input  wire        axil_wvalid,
    output reg         axil_wready,
    output reg  [1:0]  axil_bresp,
    output reg         axil_bvalid,
    input  wire        axil_bready,

    input  wire [7:0]  axil_araddr,
    input  wire        axil_arvalid,
    output reg         axil_arready,
    output reg  [31:0] axil_rdata,
    output reg  [1:0]  axil_rresp,
    output reg         axil_rvalid,
    input  wire        axil_rready,

    // ── External BRAM ports (CSR lattice storage) ─────────────────────────
    output wire [COL_ADDR_W-1:0] row_ptr_addr,
    input  wire [COL_ADDR_W-1:0] row_ptr_rdata,
    output wire [COL_ADDR_W-1:0] col_idx_addr,
    input  wire [COL_IDX_W-1:0]  col_idx_rdata,
    output wire [COL_ADDR_W-1:0] values_addr,
    input  wire [1:0]             values_rdata,

    // ── Coarse BRAM (centroid vectors, internal BRAM modelled here) ───────
    // In a real FPGA impl, this would be a separate BRAM block;
    // for simulation we leave it as a tie-off and describe the intent.
    // output wire [7:0]  coarse_bram_addr,   // hook for future expansion
    // input  wire [15:0] coarse_bram_rdata,

    // ── Status outputs (visible to PCIe DMA engine) ───────────────────────
    output wire        busy_out,
    output wire        done_out,

    // ── Inter-card sideband (would be PCIe P2P in real HW; here: wires) ──
    // This card's coarse winner, broadcast to host for arbitration:
    output reg  [15:0] coarse_top_score,
    output reg  [7:0]  coarse_top_bucket,
    // Host writes winner_bucket before issuing fine-stage start:
    input  wire [7:0]  winner_bucket_in,
    input  wire        winner_valid_in    // host asserts when fine stage ready
);

    // =========================================================================
    // Internal registers (AXI-Lite map)
    // =========================================================================
    reg [7:0]  reg_ctrl;            // 0x00
    reg [7:0]  reg_probe_addr;      // 0x08
    reg [31:0] reg_probe_data;      // 0x0C  — write triggers probe_wr_en
    reg [7:0]  reg_winner_bucket;   // 0x24
    reg        reg_boot_en;         // 0x28[0]
    reg        boot_done_sticky;
    reg        boot_error_sticky;
    reg        boot_busy;
    reg        boot_active_mem;

    wire ctrl_start        = reg_ctrl[0];
    wire ctrl_sw_rst       = reg_ctrl[1];
    wire [1:0] stage_sel   = reg_ctrl[3:2];
    wire [1:0] card_id     = reg_ctrl[5:4];
    wire ctrl_is_winner    = reg_ctrl[6];
    wire ctrl_coarse_bcast = reg_ctrl[7];

    // =========================================================================
    // Optional boot shadow memories (loaded from mock ReRAM in simulation)
    // =========================================================================
    reg [COL_ADDR_W-1:0] row_ptr_boot [0:NUM_ROWS];
    reg [COL_IDX_W-1:0]  col_idx_boot [0:COL_MEM_DEPTH-1];
    reg [1:0]            values_boot  [0:COL_MEM_DEPTH-1];

    wire [COL_ADDR_W-1:0] row_ptr_rdata_core;
    wire [COL_IDX_W-1:0]  col_idx_rdata_core;
    wire [1:0]            values_rdata_core;
    wire use_boot_mem = boot_active_mem && reg_boot_en && boot_done_sticky && !boot_error_sticky;

    assign row_ptr_rdata_core = use_boot_mem ? row_ptr_boot[row_ptr_addr] : row_ptr_rdata;
    assign col_idx_rdata_core = (use_boot_mem && (col_idx_addr < COL_MEM_DEPTH)) ? col_idx_boot[col_idx_addr] : col_idx_rdata;
    assign values_rdata_core  = (use_boot_mem && (values_addr  < COL_MEM_DEPTH)) ? values_boot[values_addr]  : values_rdata;

    // =========================================================================
    // Boot loader FSM: reads sequential 32-bit words from ReRAM and populates
    // row_ptr_boot/col_idx_boot/values_boot with a CSR seed lattice.
    //
    // Mock ReRAM contract (simulation):
    // - `boot_mem_req` high in cycle N causes the mock to produce `boot_mem_valid`
    //   and corresponding `boot_mem_rdata` in cycle N+1.
    // - Therefore each BOOT_* sequence is split into:
    //     * BOOT_REQ_*  : assert req and set address for the upcoming read
    //     * BOOT_WAIT_* : wait for `boot_mem_valid` and only then latch data
    // This avoids latching data from a different address due to addr/valid edge timing.
    // =========================================================================
    localparam BOOT_IDLE          = 4'd0;
    localparam BOOT_REQ_HDR       = 4'd1;
    localparam BOOT_WAIT_HDR      = 4'd2;
    localparam BOOT_REQ_ROW_COUNT = 4'd3;
    localparam BOOT_WAIT_ROW_COUNT= 4'd4;
    localparam BOOT_REQ_COL       = 4'd5;
    localparam BOOT_WAIT_COL      = 4'd6;
    localparam BOOT_REQ_VAL       = 4'd7;
    localparam BOOT_WAIT_VAL      = 4'd8;
    localparam BOOT_FINISH        = 4'd9;
    localparam BOOT_FAIL          = 4'd10;

    reg [3:0]  boot_state;
    reg [31:0] boot_word_addr;
    reg [31:0] boot_timeout_ctr;
    reg [31:0] boot_num_rows;
    reg [31:0] boot_row_idx;
    reg [31:0] boot_row_nnz;
    reg [31:0] boot_pair_idx;
    reg [31:0] boot_nnz_cursor;
    reg [31:0] boot_col_latch;

    reg        boot_mem_req;
    reg [31:0] boot_mem_addr;
    wire [31:0] boot_mem_rdata;
    wire        boot_mem_valid;

`ifndef SYNTHESIS
    mock_reram u_mock_reram (
        .clk   (clk),
        .rst_n (rst_n),
        .req   (boot_mem_req),
        .addr  (boot_mem_addr),
        .rdata (boot_mem_rdata),
        .valid (boot_mem_valid)
    );
`else
    assign boot_mem_rdata = 32'h0;
    assign boot_mem_valid = 1'b0;
`endif

    function [1:0] boot_enc_val;
        input [31:0] w;
        begin
            case (w[7:0])
                8'h01: boot_enc_val = 2'b01;
                8'hFF: boot_enc_val = 2'b10;
                default: boot_enc_val = 2'b00;
            endcase
        end
    endfunction

    // =========================================================================
    // AXI-Lite write channel
    // =========================================================================
    reg        axil_aw_pend;
    reg [7:0]  axil_aw_latch;
    reg        probe_wr_en_r;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axil_awready  <= 1'b1;
            axil_wready   <= 1'b1;
            axil_bvalid   <= 1'b0;
            axil_bresp    <= 2'b00;
            axil_aw_pend  <= 1'b0;
            axil_aw_latch <= 8'h00;
            probe_wr_en_r <= 1'b0;
            reg_ctrl         <= 8'h00;
            reg_probe_addr   <= 8'h00;
            reg_probe_data   <= 32'h0;
            reg_winner_bucket<= 8'h00;
            reg_boot_en       <= 1'b0;
        end else begin
            probe_wr_en_r <= 1'b0;

            // Address phase
            if (axil_awvalid && axil_awready) begin
                axil_aw_latch <= axil_awaddr;
                axil_aw_pend  <= 1'b1;
                axil_awready  <= 1'b0;
            end

            // Data phase + register write
            if (axil_wvalid && axil_wready && axil_aw_pend) begin
                axil_wready  <= 1'b0;
                axil_aw_pend <= 1'b0;
                axil_awready <= 1'b1;
                axil_bvalid  <= 1'b1;
                axil_bresp   <= 2'b00;
                case (axil_aw_latch[7:0])
                    8'h00: reg_ctrl          <= axil_wdata[7:0];
                    8'h08: reg_probe_addr    <= axil_wdata[7:0];
                    8'h0C: begin
                               reg_probe_data <= axil_wdata;
                               probe_wr_en_r  <= 1'b1;
                           end
                    8'h24: reg_winner_bucket <= axil_wdata[7:0];
                    8'h28: reg_boot_en        <= axil_wdata[0];
                    default: ; // read-only registers ignored
                endcase
            end else if (!axil_aw_pend) begin
                axil_wready <= 1'b1;
            end

            if (axil_bvalid && axil_bready) begin
                axil_bvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // AXI-Lite read channel
    // =========================================================================
    wire       core_busy, core_done, core_result_rdy;
    wire signed [15:0] core_result_score;
    wire [ROW_ADDR_W-1:0] core_result_row;
    wire [7:0]            core_result_idx;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            axil_arready <= 1'b1;
            axil_rvalid  <= 1'b0;
            axil_rresp   <= 2'b00;
            axil_rdata   <= 32'h0;
        end else begin
            if (axil_arvalid && axil_arready) begin
                axil_arready <= 1'b0;
                axil_rvalid  <= 1'b1;
                axil_rresp   <= 2'b00;
                case (axil_araddr[7:0])
                    8'h00: axil_rdata <= {24'h0, reg_ctrl};
                    8'h04: axil_rdata <= {29'h0, core_result_rdy, core_done, core_busy};
                    8'h08: axil_rdata <= {24'h0, reg_probe_addr};
                    8'h0C: axil_rdata <= reg_probe_data;
                    8'h10: axil_rdata <= {{16{core_result_score[15]}}, core_result_score};
                    8'h14: axil_rdata <= {{(32-ROW_ADDR_W){1'b0}}, core_result_row};
                    8'h18: axil_rdata <= {24'h0, core_result_idx};
                    8'h1C: axil_rdata <= {16'h0, coarse_top_score};
                    8'h20: axil_rdata <= {24'h0, coarse_top_bucket};
                    8'h24: axil_rdata <= {24'h0, reg_winner_bucket};
                    8'h28: axil_rdata <= {29'h0, boot_error_sticky, boot_done_sticky, reg_boot_en};
                    default: axil_rdata <= 32'hDEAD_BEEF;
                endcase
            end else if (axil_rvalid && axil_rready) begin
                axil_rvalid  <= 1'b1;  // keep valid until accepted
                axil_arready <= 1'b1;
                if (axil_rready) axil_rvalid <= 1'b0;
            end
        end
    end

    // =========================================================================
    // Hierarchy controller FSM
    // =========================================================================
    // Stage encoding
    localparam STAGE_COARSE = 2'b00;
    localparam STAGE_FINE   = 2'b01;
    localparam STAGE_FULL   = 2'b10;

    localparam HS_IDLE      = 3'd0;
    localparam HS_COARSE    = 3'd1;
    localparam HS_COARSE_W  = 3'd2;  // wait coarse done
    localparam HS_FINE      = 3'd3;
    localparam HS_FINE_W    = 3'd4;  // wait fine done
    localparam HS_DONE      = 3'd5;

    reg [2:0]  h_state;
    reg        core_start_r;
    reg [7:0]  coarse_ptr;           // centroid row counter for coarse scan
    reg signed [15:0] coarse_best;
    reg [7:0]  coarse_best_bucket;

    // Row slice for this card (global CSR row indices)
    localparam integer ROWS_PER_CARD_I = NUM_ROWS / NUM_CARDS;
    localparam integer ROW_SCAN_FIRST_I = ROWS_PER_CARD_I * CARD_ID;
    wire [ROW_ADDR_W-1:0] pcie_row_first = ROW_SCAN_FIRST_I[ROW_ADDR_W-1:0];
    wire [ROW_ADDR_W:0]   pcie_row_len   = ROWS_PER_CARD_I[ROW_ADDR_W:0];

    // Start pulse edge-detect
    reg ctrl_start_prev;
    wire start_pulse = ctrl_start && !ctrl_start_prev;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            h_state          <= HS_IDLE;
            core_start_r     <= 1'b0;
            ctrl_start_prev  <= 1'b0;
            coarse_ptr       <= 8'h00;
            coarse_best      <= 16'sh8000;
            coarse_best_bucket <= 8'h00;
            coarse_top_score  <= 16'h0;
            coarse_top_bucket <= 8'h00;
            boot_done_sticky  <= 1'b0;
            boot_error_sticky <= 1'b0;
            boot_busy         <= 1'b0;
            boot_active_mem   <= 1'b0;
            boot_state        <= BOOT_IDLE;
            boot_mem_req      <= 1'b0;
            boot_mem_addr     <= 32'h0;
            boot_word_addr    <= 32'h0;
            boot_timeout_ctr  <= 32'h0;
            boot_num_rows     <= 32'h0;
            boot_row_idx      <= 32'h0;
            boot_row_nnz      <= 32'h0;
            boot_pair_idx     <= 32'h0;
            boot_nnz_cursor   <= 32'h0;
            boot_col_latch    <= 32'h0;
        end else begin
            ctrl_start_prev <= ctrl_start;
            core_start_r    <= 1'b0;

            case (h_state)
                HS_IDLE: begin
                    if (start_pulse) begin
                        coarse_best       <= 16'sh8000;
                        coarse_best_bucket<= 8'h00;
                        coarse_ptr        <= 8'h00;
                        if (stage_sel == STAGE_FINE)
                            // Fine-only: every card runs its row shard in parallel.
                            h_state <= HS_FINE;
                        else begin
                            // Coarse or full: all cards run coarse simultaneously.
                            h_state <= HS_COARSE;
                        end
                    end
                end

                // ── Coarse stage: iterate centroid BRAMs ──────────────────
                // In real silicon each centroid comparison = 1 cycle;
                // here we model it as a pass-through; host scores via STATUS reads.
                HS_COARSE: begin
                    // Dummy: set synthetic coarse score for simulation visibility
                    coarse_top_score  <= 16'h0400 + {8'h0, coarse_ptr};
                    coarse_top_bucket <= coarse_ptr;
                    h_state <= HS_COARSE_W;
                end

                HS_COARSE_W: begin
                    if (stage_sel == STAGE_COARSE)
                        h_state <= HS_DONE;
                    else
                        // FULL: all cards enter fine on their row shard (no single winner).
                        h_state <= HS_FINE;
                end

                // ── Fine stage: rshl_core over this card's row range ─────────
                HS_FINE: begin
                    core_start_r <= 1'b1;
                    h_state      <= HS_FINE_W;
                end

                HS_FINE_W: begin
                    if (core_done)
                        h_state <= HS_DONE;
                end

                HS_DONE: begin
                    h_state <= HS_IDLE;
                end

                default: h_state <= HS_IDLE;
            endcase

            // Synchronous reset overrides
            if (ctrl_sw_rst) begin
                h_state      <= HS_IDLE;
                core_start_r <= 1'b0;
            end

            // Boot control clears completion flags on disable.
            if (!reg_boot_en) begin
                if (!boot_busy) begin
                    boot_done_sticky  <= 1'b0;
                    boot_error_sticky <= 1'b0;
                    boot_active_mem   <= 1'b0;
                end
            end

            // -------------------- boot FSM --------------------
            boot_mem_req <= 1'b0;
            if (!reg_boot_en) begin
                boot_state <= BOOT_IDLE;
                boot_busy  <= 1'b0;
            end else begin
                case (boot_state)
                    BOOT_IDLE: begin
                        if (!boot_done_sticky && !boot_error_sticky) begin
                            boot_busy        <= 1'b1;
                            boot_word_addr   <= 32'd0;
                            boot_timeout_ctr <= 32'd0;
                            boot_row_idx     <= 32'd0;
                            boot_nnz_cursor  <= 32'd0;
                            boot_mem_addr    <= 32'd0;
                            boot_state       <= BOOT_REQ_HDR;
                        end
                    end
                    BOOT_REQ_HDR: begin
                        boot_mem_req       <= 1'b1;
                        boot_mem_addr      <= boot_word_addr;
                        boot_timeout_ctr  <= 32'd0;
                        boot_state         <= BOOT_WAIT_HDR;
                    end
                    BOOT_WAIT_HDR: begin
                        if (boot_mem_valid) begin
                            boot_num_rows  <= boot_mem_rdata;
                            boot_word_addr <= 32'd1; // first row count word
                            if (boot_mem_rdata > NUM_ROWS) begin
                                boot_state <= BOOT_FAIL;
                            end else begin
                                boot_state <= BOOT_REQ_ROW_COUNT;
                            end
                        end else if (boot_timeout_ctr >= BOOT_TIMEOUT_CYC) begin
                            boot_state <= BOOT_FAIL;
                        end else begin
                            boot_timeout_ctr <= boot_timeout_ctr + 1'b1;
                        end
                    end
                    BOOT_REQ_ROW_COUNT: begin
                        if (boot_row_idx >= boot_num_rows) begin
                            row_ptr_boot[boot_num_rows[ROW_ADDR_W-1:0]] <= boot_nnz_cursor[COL_ADDR_W-1:0];
                            boot_state <= BOOT_FINISH;
                        end else begin
                            row_ptr_boot[boot_row_idx[ROW_ADDR_W-1:0]] <= boot_nnz_cursor[COL_ADDR_W-1:0];
                            boot_mem_addr <= boot_word_addr;
                            boot_mem_req <= 1'b1;
                            boot_timeout_ctr <= 32'd0;
                            boot_state <= BOOT_WAIT_ROW_COUNT;
                        end
                    end
                    BOOT_WAIT_ROW_COUNT: begin
                        if (boot_mem_valid) begin
                            boot_row_nnz <= boot_mem_rdata;
                            boot_word_addr <= boot_word_addr + 1'b1;
                            boot_pair_idx <= 32'd0;
                            if ((boot_nnz_cursor + boot_mem_rdata) > COL_MEM_DEPTH) begin
                                boot_state <= BOOT_FAIL;
                            end else begin
                                boot_state <= BOOT_REQ_COL;
                            end
                        end else if (boot_timeout_ctr >= BOOT_TIMEOUT_CYC) begin
                            boot_state <= BOOT_FAIL;
                        end else begin
                            boot_timeout_ctr <= boot_timeout_ctr + 1'b1;
                        end
                    end
                    BOOT_REQ_COL: begin
                        if (boot_pair_idx >= boot_row_nnz) begin
                            boot_nnz_cursor <= boot_nnz_cursor + boot_row_nnz;
                            boot_row_idx <= boot_row_idx + 1'b1;
                            boot_state <= BOOT_REQ_ROW_COUNT;
                        end else begin
                            boot_mem_addr <= boot_word_addr;
                            boot_mem_req <= 1'b1;
                            boot_timeout_ctr <= 32'd0;
                            boot_state <= BOOT_WAIT_COL;
                        end
                    end
                    BOOT_WAIT_COL: begin
                        if (boot_mem_valid) begin
                            boot_col_latch <= boot_mem_rdata;
                            boot_timeout_ctr <= 32'd0;
                            boot_state <= BOOT_REQ_VAL;
                        end else if (boot_timeout_ctr >= BOOT_TIMEOUT_CYC) begin
                            boot_state <= BOOT_FAIL;
                        end else begin
                            boot_timeout_ctr <= boot_timeout_ctr + 1'b1;
                        end
                    end
                    BOOT_REQ_VAL: begin
                        boot_mem_req      <= 1'b1;
                        boot_mem_addr     <= boot_word_addr + 1'b1;
                        boot_timeout_ctr <= 32'd0;
                        boot_state        <= BOOT_WAIT_VAL;
                    end
                    BOOT_WAIT_VAL: begin
                        if (boot_mem_valid) begin
                            col_idx_boot[(boot_nnz_cursor + boot_pair_idx)] <= boot_col_latch[COL_IDX_W-1:0];
                            values_boot[(boot_nnz_cursor + boot_pair_idx)]  <= boot_enc_val(boot_mem_rdata);
                            boot_pair_idx <= boot_pair_idx + 1'b1;
                            boot_word_addr <= boot_word_addr + 2;
                            boot_state <= BOOT_REQ_COL;
                        end else if (boot_timeout_ctr >= BOOT_TIMEOUT_CYC) begin
                            boot_state <= BOOT_FAIL;
                        end else begin
                            boot_timeout_ctr <= boot_timeout_ctr + 1'b1;
                        end
                    end
                    BOOT_FINISH: begin
                        boot_busy         <= 1'b0;
                        boot_done_sticky  <= 1'b1;
                        boot_error_sticky <= 1'b0;
                        boot_active_mem   <= 1'b1;
                        boot_state        <= BOOT_IDLE;
                    end
                    BOOT_FAIL: begin
                        boot_busy         <= 1'b0;
                        boot_done_sticky  <= 1'b0;
                        boot_error_sticky <= 1'b1;
                        boot_active_mem   <= 1'b0;
                        boot_state        <= BOOT_IDLE;
                    end
                    default: boot_state <= BOOT_IDLE;
                endcase
            end
        end
    end

    // =========================================================================
    // rshl_core instantiation (fine-grained search engine)
    // =========================================================================
    rshl_core #(
        .DIM           (DIM),
        .NUM_ROWS      (NUM_ROWS),
        .TOP_K         (TOP_K),
        .PROBE_NNZ_MAX (PROBE_NNZ_MAX),
        .COL_MEM_DEPTH (COL_MEM_DEPTH),
        .COL_IDX_W     (COL_IDX_W),
        .ROW_ADDR_W    (ROW_ADDR_W),
        .COL_ADDR_W    (COL_ADDR_W),
        .PROBE_ADDR_W  (PROBE_ADDR_W)
    ) u_rshl_core (
        .clk            (clk),
        .rst_n          (rst_n && !ctrl_sw_rst),
        .row_scan_first (pcie_row_first),
        .row_scan_len   (pcie_row_len),
        .row_ptr_addr   (row_ptr_addr),
        .row_ptr_rdata  (row_ptr_rdata_core),
        .col_idx_addr   (col_idx_addr),
        .col_idx_rdata  (col_idx_rdata_core),
        .values_addr    (values_addr),
        .values_rdata   (values_rdata_core),
        .probe_addr     (reg_probe_addr[PROBE_ADDR_W-1:0]),
        .probe_wr_en    (probe_wr_en_r),
        .probe_data     (reg_probe_data),
        .start          (core_start_r),
        .busy           (core_busy),
        .done           (core_done),
        .result_rdy     (core_result_rdy),
        .result_score   (core_result_score),
        .result_row     (core_result_row),
        .result_idx     (core_result_idx)
    );

    assign busy_out = core_busy | (h_state != HS_IDLE && h_state != HS_DONE);
    assign done_out = (h_state == HS_DONE);

endmodule

`default_nettype wire
