function(target_add_warnings target)

  if(MSVC)
    target_compile_options(${target} PRIVATE)
  else()
    target_compile_options(${target} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-pedantic -Wall -Wextra -Wno-sign-compare -Wno-unused-variable -Wno-unused-parameter>)
  endif()
endfunction()
