# Contributing
__We appreciate all kinds of help, so thank you!__


## Code contribution guide
This guide is for those who want to extend the module or documentation. If you just want to use the package, read [this other guide](./docs/2-reference_guide/reference_guide.md) instead.

Code in this repository should conform to PEP8 standards. Style/lint checks are run to validate this. Line length must be limited to no more than 100 characters.

### Initial set-up and installing dependencies
In order to contribute, you will need to [install the module from source](./docs/2-reference_guide/reference_guide.md#installation-from-source). If you do not have write permissions to the original Entanglement Forging repo, you will need to fork it to your personal account first, and submit all Pull Requests (PR) from there. Even if you have write permissions, forking the repo should always work; so this is the recommended approach.

### Running tests
If you haven't, install all packages requiered for development:
```
pip install -r requirements-dev.txt
```

To run tests:
```
tox -e{env}
```
where you replace `{env}` with `py37`, `py38` or `py39` depending on which version of python you have (to check python version, type `python --version` in the terminal).

To run linting tests (checks formatting/syntax):
```
tox -elint
```

To run all the tests:
```
python -m unittest discover -v tests
```

To run notebook tests (for more info, see [here](https://github.com/ReviewNB/treon):
```
pip install pytest nbmake
treon docs/
```
Note: notebook tests check for execution and time-out errors, not correctness.

### Making a pull request

#### Step-by-step
1. To make a contribution, first set up a remote branch (here called `my-contribution`) either in your fork (i.e. `origin`) or the original repo (i.e. `upstream`). In the absence of a fork, the (only) remote will simply be referred to all the time by the name `origin` (i.e. replace `upstream` in all commands):
   ```
   git checkout main
   git pull origin
   git checkout -b my-contribution
   ```
   ... make your contribution now (edit some code, add some files) ...
   ```
   git add .
   git commit -m 'initial working version of my contribution'
   git push -u origin my-contribution
   ```
2. Before making a Pull Request always get the latest changes from `main` (`upstream` if there is a fork, `origin` otherwise):
   ```
   git checkout main
   git pull upstream
   git checkout my-contribution
   git merge main
   ```
   ... fix any merge conflicts here ...
   ```
   git add .
   git commit -m 'merged updates from main'
   git push
   ```
3. Go back to the appropriate Entanglement Forging repo on GitHub (i.e. fork or original), switch to your contribution branch (same name: `my-contribution`), and click _"Pull Request"_. Write a clear explanation of the feature.
4. Under _Reviewer_, select Aggie Branczyk __and__ Caleb Johnson.
5. Click _"Create Pull Request"_.
6. Your Pull Request will be reviewed and, if everything is ok, it will be merged.

#### Pull request checklist
When submitting a pull request and you feel it is ready for review, please ensure that:
1. The code follows the _code style_ of this project and successfully passes the _unit tests_. Entanglement Forging uses [Pylint](https://www.pylint.org) and [PEP8](https://www.python.org/dev/peps/pep-0008) style guidelines. For this you can run:
   ```
   tox -elint
   ```


## Other ways of contributing
Other than submitting new sourcecode, users can contribute in the following meaningful ways:
 - __Reporting Bugs and Requesting Features__: Users are encouraged to use Github Issues for reporting issues are requesting features.
 - __Ask/Answer Questions and Discuss Entanglement Forging__: Users are encouraged to use Github Discussions for engaging with researchers, developers, and other users regarding this project.
