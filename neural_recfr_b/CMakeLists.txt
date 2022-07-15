find_package(OpenMP REQUIRED)


set(TENSORFLOW_DIR ../third_party/tensorflow)
# link dynamic tensorflow lib.
include_directories( 
  ${CMAKE_CURRENT_SOURCE_DIR}
  ${TENSORFLOW_DIR}/include
  ${TENSORFLOW_DIR}/bazel_include
  ${TENSORFLOW_DIR}/include/nsync/public
  ${TENSORFLOW_DIR}/include/com_google_protobuf/src
) 
link_directories(${TENSORFLOW_DIR}/bin)

add_library (neural_base OBJECT
  device_manager.h
  vpevaluator.h
  vpevaluator.cc
  vpnet.h
  vpnet.cc
  play.h
  play.cc
  local_best_response.h
  local_best_response.cc
  universal_poker_exploitability.h
  universal_poker_exploitability.cc
)
target_include_directories (neural_base PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})
target_link_libraries(neural_base PUBLIC tensorflow_cc OpenMP::OpenMP_CXX)

add_executable(run_neural_recfr_b neural_recfr_b.cc run_deep_cfr.cc  deep_cfr.h 
    neural_recfr_b_solver.cc  neural_recfr_b_solver.h
    ${OPEN_SPIEL_OBJECTS})
target_link_libraries(run_neural_recfr_b neural_base)