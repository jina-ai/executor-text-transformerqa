def pytest_addoption(parser):
    parser.addoption(
        '--gpu', action='store_true', dest='gpu', default=False, help='enable gpu tests'
    )


def pytest_configure(config):
    # only enable gpu test if --gpu flag is added to pytest command
    if not config.option.gpu:
        setattr(config.option, 'markexpr', 'not gpu')
