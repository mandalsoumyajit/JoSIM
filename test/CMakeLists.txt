# Commad line options

#add_integration_test(
#  NAME output_after_cir
#  CIR basic_jtl.cir
#  OUT basic_jtl.csv
#  OVERWRITE_ARGS basic_jtl.cir -o basic_jtl.csv
#)

# Tests

add_integration_test(
  NAME test_capacitor
  CIR comp/capacitor.cir
)

add_integration_test(
  NAME test_cccs
  CIR comp/cccs.cir
)

add_integration_test(
  NAME test_ccvs
  CIR comp/ccvs.cir
)

add_integration_test(
  NAME test_inductor
  CIR comp/inductor.cir
)

add_integration_test(
  NAME test_jj
  CIR comp/jj.cir
)

add_integration_test(
  NAME test_ps
  CIR comp/ps.cir
)

add_integration_test(
  NAME test_resistor
  CIR comp/resistor.cir
)

add_integration_test(
  NAME test_tx
  CIR comp/tx.cir
)

add_integration_test(
  NAME test_vccs
  CIR comp/vccs.cir
)

add_integration_test(
  NAME test_vcvs
  CIR comp/vcvs.cir
)

add_integration_test(
  NAME test_vs
  CIR comp/vs.cir
)

add_integration_test(
  NAME ex_dcsfq_jtl_sink
  CIR ex_dcsfq_jtl_sink.cir
  OUT ex_dcsfq_jtl_sink.csv
)

add_integration_test(
  NAME ex_gen_pp_ipht
  CIR ex_gen_pp_ipht.cir
  OUT ex_gen_pp_ipht.csv
)

add_integration_test(
  NAME ex_jtl_basic
  CIR ex_jtl_basic.cir
  OUT ex_jtl_basic.csv
)

add_integration_test(
  NAME ex_jtl_string
  CIR ex_jtl_string.cir
  OUT ex_jtl_string.csv
)

add_integration_test(
  NAME ex_ksa4bit
  CIR ex_ksa4bit.cir
  OUT ex_ksa4bit.csv
)

add_integration_test(
  NAME ex_mitll_dff_wr
  CIR ex_mitll_dff_wr.cir
  OUT ex_mitll_dff_wr.csv
)

add_integration_test(
  NAME ex_mitll_xor
  CIR ex_mitll_xor_wr.cir
  OUT ex_mitll_xor_wr.csv
)

add_integration_test(
  NAME test_param_nest
  CIR param/test_param_nest.cir
)

add_integration_test(
  NAME test_param_undef
  CIR param/test_param_undef.cir
  WILL_FAIL
)

add_integration_test(
  NAME test_line_cont
  CIR syntax/test_line_cont.cir
)

add_integration_test(
  NAME test_line_comment
  CIR syntax/test_line_comment.cir
)

add_integration_test(
  NAME test_output
  CIR syntax/test_output.cir
)


add_test(
  NAME "integration::margins_dff"
  COMMAND $<TARGET_FILE:josim-cli> -M "@DEFAULT|X0"
          "--margin-file=${PROJECT_BINARY_DIR}/tests/integration/dff_margins.csv"
          "${PROJECT_SOURCE_DIR}/test/ex_mitll_dff_wr.cir"
)
set_tests_properties("integration::margins_dff" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Critical Operating Margin Report.*@BIAS\\|X0"
)

add_test(
  NAME "integration::margins_devices_jtl"
  COMMAND $<TARGET_FILE:josim-cli> -M "@DEVICES"
          "${PROJECT_SOURCE_DIR}/test/ex_jtl_basic.cir"
)
set_tests_properties("integration::margins_devices_jtl" PROPERTIES
  PASS_REGULAR_EXPRESSION "Critical Bottleneck"
)

add_test(
  NAME "integration::verify_events_export"
  COMMAND $<TARGET_FILE:josim-cli> -e "${PROJECT_BINARY_DIR}/tests/integration/dff_golden.events.csv"
          "${PROJECT_SOURCE_DIR}/test/ex_mitll_dff_wr.cir"
)

add_test(
  NAME "integration::verify_events_check"
  COMMAND $<TARGET_FILE:josim-cli>
          "--verify-events=${PROJECT_BINARY_DIR}/tests/integration/dff_golden.events.csv"
          "${PROJECT_SOURCE_DIR}/test/ex_mitll_dff_wr.cir"
)
set_tests_properties("integration::verify_events_check" PROPERTIES
  DEPENDS "integration::verify_events_export"
  PASS_REGULAR_EXPRESSION "JoSIM Event Verification: PASS"
)

add_test(
  NAME "integration::test_sfq_assert"
  COMMAND $<TARGET_FILE:josim-cli>
          "${PROJECT_SOURCE_DIR}/test/syntax/test_sfq_assert.cir"
)
set_tests_properties("integration::test_sfq_assert" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Assertions: PASS \\(5/5 assertions passed\\)"
)

add_test(
  NAME "integration::test_pat_assert"
  COMMAND $<TARGET_FILE:josim-cli>
          "${PROJECT_SOURCE_DIR}/test/syntax/test_pat_assert.cir"
)
set_tests_properties("integration::test_pat_assert" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Assertions: PASS \\(2/2 assertions passed\\)"
)

add_test(
  NAME "integration::cli_assert_dff"
  COMMAND $<TARGET_FILE:josim-cli>
          -A "B2|X3:COUNT=2"
          -A "B2|X3:TIMES=108.8p,248.9p TOL=0.5p"
          -A "B2|X3:TD=100p PER=20p BITS=100000010000"
          -A "B2|X3:AFTER=B2|X4 DELAY=47.5p TOL=3p"
          "${PROJECT_SOURCE_DIR}/test/ex_mitll_dff_wr.cir"
)
set_tests_properties("integration::cli_assert_dff" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Assertions: PASS \\(4/4 assertions passed\\)"
)

add_test(
  NAME "integration::test_subckt_math_glob"
  COMMAND $<TARGET_FILE:josim-cli>
          "${PROJECT_SOURCE_DIR}/test/param/test_subckt_math_glob.cir"
)
set_tests_properties("integration::test_subckt_math_glob" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Assertions: PASS \\(9/9 assertions passed\\)"
)

add_test(
  NAME "integration::test_meas"
  COMMAND $<TARGET_FILE:josim-cli>
          "--meas-file=${PROJECT_BINARY_DIR}/tests/integration/meas_report.csv"
          "${PROJECT_SOURCE_DIR}/test/syntax/test_meas.cir"
)
set_tests_properties("integration::test_meas" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Measurement Report \\(16 measurement\\(s\\)\\).*T_PROP.*FLUX_NORM"
)

add_test(
  NAME "integration::test_ic_flux_loops"
  COMMAND $<TARGET_FILE:josim-cli>
          "--check-loops=${PROJECT_BINARY_DIR}/tests/integration/loop_audit.csv"
          "${PROJECT_SOURCE_DIR}/test/syntax/test_ic_flux_loops.cir"
)
set_tests_properties("integration::test_ic_flux_loops" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Superconducting Loop \\(beta_L\\) Audit Report \\(3 loops, 1 storage, 0 degenerate\\)"
)

add_test(
  NAME "integration::test_tx_loss"
  COMMAND $<TARGET_FILE:josim-cli>
          "${PROJECT_SOURCE_DIR}/test/syntax/test_tx_loss.cir"
)
set_tests_properties("integration::test_tx_loss" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Assertions: PASS \\(2/2 assertions passed\\).*JoSIM Measurement Report \\(9 measurement\\(s\\)\\)"
)

add_test(
  NAME "integration::test_jitter_ber"
  COMMAND $<TARGET_FILE:josim-cli>
          "--jitter-file=${PROJECT_BINARY_DIR}/tests/integration/jitter_report.json"
          "${PROJECT_SOURCE_DIR}/test/syntax/test_jitter_ber.cir"
)
set_tests_properties("integration::test_jitter_ber" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Timing Jitter, Skew & BER Report \\(5 target\\(s\\), 1 trial\\(s\\)\\).*J_LOSS\\(B11\\).*3/4.*2\\.50e-01.*PASS.*JoSIM Measurement Report \\(6 measurement\\(s\\)\\)"
)

add_test(
  NAME "integration::test_iv_squid"
  COMMAND $<TARGET_FILE:josim-cli>
          "${PROJECT_SOURCE_DIR}/test/syntax/test_iv_squid.cir"
)
set_tests_properties("integration::test_iv_squid" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM I-V & Shapiro-Step Characterization Report.*JoSIM SQUID Characterization Report.*JoSIM Measurement Report \\(13 measurement\\(s\\)\\)"
)

find_package(Python3 COMPONENTS Interpreter QUIET)
if(Python3_Interpreter_FOUND)
  execute_process(
    COMMAND ${Python3_EXECUTABLE} -c "import numpy"
    RESULT_VARIABLE _PYTHON_NO_NUMPY
    OUTPUT_QUIET
    ERROR_QUIET
  )
endif()
if(Python3_Interpreter_FOUND AND _PYTHON_NO_NUMPY EQUAL 0)
  add_test(
    NAME "integration::test_pyjosim"
    COMMAND ${Python3_EXECUTABLE}
            "${PROJECT_SOURCE_DIR}/test/test_pyjosim.py"
            $<TARGET_FILE:josim_c>
  )
  set_tests_properties("integration::test_pyjosim" PROPERTIES
    PASS_REGULAR_EXPRESSION "pyjosim PASS: 5 traces x 1199 steps, 6 SFQ events"
  )

  add_test(
    NAME "integration::test_cosim_step1"
    COMMAND ${Python3_EXECUTABLE}
            "${PROJECT_SOURCE_DIR}/test/test_cosim_step1.py"
            $<TARGET_FILE:josim_c>
  )
  set_tests_properties("integration::test_cosim_step1" PROPERTIES
    PASS_REGULAR_EXPRESSION "ALL 4 STEP 1 CO-SIMULATION BENCHMARKS PASSED!"
  )

  if(NOT WIN32)
    add_test(
      NAME "integration::test_cosim_step3_and_veriloga_comparison"
      COMMAND ${CMAKE_COMMAND} -E env
              "JOSIM_BUILD_DIR=$<TARGET_FILE_DIR:josim_c>"
              "JOSIM_C_LIB=$<TARGET_FILE:josim_c>"
              "JOSIM_SPECTRE_DPI_LIB=$<TARGET_FILE:josim_spectre_dpi>"
              "JOSIM_COSIM_SERVER_BIN=$<TARGET_FILE:josim-cosim-server>"
              ${Python3_EXECUTABLE}
              "${PROJECT_SOURCE_DIR}/test/test_cosim_step3_and_veriloga_comparison.py"
    )
    set_tests_properties("integration::test_cosim_step3_and_veriloga_comparison" PROPERTIES
      PASS_REGULAR_EXPRESSION "ALL STEP 3 & VERILOG-A COMPARISON BENCHMARKS PASSED."
    )
  endif()

  if(NOT WIN32 AND TARGET josim_cosim_cm)
    add_test(
      NAME "integration::test_cosim_simulators_and_step2"
      COMMAND ${CMAKE_COMMAND} -E env
              "JOSIM_C_LIB=$<TARGET_FILE:josim_c>"
              "JOSIM_COSIM_SERVER_BIN=$<TARGET_FILE:josim-cosim-server>"
              ${Python3_EXECUTABLE}
              "${PROJECT_SOURCE_DIR}/test/test_cosim_simulators_and_step2.py"
    )
    set_tests_properties("integration::test_cosim_simulators_and_step2" PROPERTIES
      PASS_REGULAR_EXPRESSION "All 5 real-simulator Step 1 & Step 2 co-simulation benchmarks passed!"
    )

    add_test(
      NAME "integration::test_cosim_step4_layer3b"
      COMMAND ${CMAKE_COMMAND} -E env
              "JOSIM_BUILD_DIR=$<TARGET_FILE_DIR:josim_c>"
              "JOSIM_C_LIB=$<TARGET_FILE:josim_c>"
              "JOSIM_COSIM_SERVER_BIN=$<TARGET_FILE:josim-cosim-server>"
              ${Python3_EXECUTABLE}
              "${PROJECT_SOURCE_DIR}/test/test_cosim_step4_layer3b.py"
    )
    set_tests_properties("integration::test_cosim_step4_layer3b" PROPERTIES
      PASS_REGULAR_EXPRESSION "ALL STEP 4 \\(LAYER 3B\\) CO-SIMULATION BENCHMARKS PASSED."
    )
  endif()
endif()

add_test(
  NAME "integration::test_sens"
  COMMAND $<TARGET_FILE:josim-cli>
          "--sens-file=${PROJECT_BINARY_DIR}/tests/integration/sens_report.json"
          "${PROJECT_SOURCE_DIR}/test/syntax/test_sens.cir"
)
set_tests_properties("integration::test_sens" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Batched OAAT Sensitivity & Margin Centering Report \\(9 target\\(s\\), 253 batched trials, metric = T_PROP.*JoSIM Measurement Report \\(11 measurement\\(s\\), 253 batched trials\\)"
)

add_test(
  NAME "integration::test_ntron"
  COMMAND $<TARGET_FILE:josim-cli>
          "${PROJECT_SOURCE_DIR}/test/syntax/test_ntron.cir"
)
set_tests_properties("integration::test_ntron" PROPERTIES
  PASS_REGULAR_EXPRESSION "JoSIM Assertions: PASS \\(1/1 assertions passed\\).*JoSIM Measurement Report \\(15 measurement\\(s\\)\\).*V1_PEAK.*3\\.6351e-03.*T_B1_SW.*1\\.2353e-11"
)








