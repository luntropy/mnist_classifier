import nox

@nox.session(name = 'model_test.py')
def tests(session):
    session.run('pytest')
