import nox

@nox.session(name = 'Test Project')
def tests(session):
    session.run('pytest')
