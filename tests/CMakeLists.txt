##############################################################################
#
#  xad-forge tests
#
##############################################################################

include(FetchContent)

FetchContent_Declare(
    googletest
    GIT_REPOSITORY https://github.com/google/googletest.git
    GIT_TAG v1.14.0
)
FetchContent_MakeAvailable(googletest)

add_executable(xad-forge-tests
    forgebackend_test.cpp
)

target_link_libraries(xad-forge-tests PRIVATE
    xad-forge
    GTest::gtest
    GTest::gtest_main
)

include(GoogleTest)
gtest_discover_tests(xad-forge-tests)
