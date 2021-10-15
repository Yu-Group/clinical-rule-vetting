def pytest_addoption(parser):
    parser.addoption("--project", action="store", default='None')


def pytest_generate_tests(metafunc):
    # This is called for every test. Only get/set command line arguments
    # if the argument is specified in the list of test "fixturenames".
    option_value = metafunc.config.option.project
    if 'project' in metafunc.fixturenames and option_value is not None:
        metafunc.parametrize("project", [option_value])
