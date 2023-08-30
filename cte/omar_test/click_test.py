import click

@click.command()
# @click.option('--count', default=1, help='Number of greetings.')
@click.option('--name', prompt='Your name',
              help='The person to greet.')
def hello(name):
    """Simple program that greets NAME"""
    click.echo(f"Hello {name}!")


@click.command()
@click.option('-r','--repository', prompt='Connect to repository',
              help='The repository you would like to connect to on your github')
def connect(repo_name):
    """Simple program that git adds file"""
    click.echo(f"This is the repo that you are connecting to: {repo_name}!")


if __name__ == '__main__':
    add_file()



@click.command()
# @click.option('--count', default=1, help='Number of greetings.')
@click.option('-af','--add_file', prompt='Add the file name that you want to add to your github',
              help='The file on your local host that you want to commit to github')
def add_file(file):
    """Simple program that git adds file"""
    click.echo(f"This is the file that you are adding {file}!")
    !git add file

if __name__ == '__main__':
    add_file()
