# Commonsense Reasoning

The goal of this project was to explore modern neural network technology in the application of
discerning and generating statements that are ‘reasonable’, in what is known as commonsense
reasoning. We built off of the work of Saeedi et al. In their work on the 2020 SemEval task,
Commonsense Validation and Explanation (ComVE). SemEval is a workshop that creates a
variety of semantic evaluation tasks to examine the state of the art in the practical application of
natural language processing. This particular task involved three sections: task A, Validation, in
which a program tries to select which of two statements is more sensical; task B, Explanation, in
which the program is given an illogical statement and has to choose between three statements to
find the one that works as the best explanation as to why the statement is illogical; and task C,
generation, in which the program must generate a novel explanation as to why a given statement
is illogical.


<br />


![Subtask A Demo](https://github.com/JustinLee38/common-sense-reasoning/blob/1c73c6be7c5d656a263f7748afb84e47ea538846/media/subtaskA.JPG)

<br />


## Link to presentation
- [Youtube](https://youtu.be/OX6A1rDsEq0) - The senior design presentation


## How to use it

```bash
$ # Get the code
$ git clone https://github.com/JustinLee38/CommonSenseReasoning.git
$ cd CommonSenseReasoning
$
$ # Virtualenv modules installation (Unix based systems)
$ virtualenv env
$ source env/bin/activate
$
$ # Virtualenv modules installation (Windows based systems)
$ # virtualenv env
$ # .\env\Scripts\activate
$
$ # Install modules - SQLite Database
$ pip3 install -r requirements.txt
$
$ # OR with PostgreSQL connector
$ # pip install -r requirements-pgsql.txt
$
$ # Set the FLASK_APP environment variable
$ (Unix/Mac) export FLASK_APP=run.py
$ (Windows) set FLASK_APP=run.py
$ (Powershell) $env:FLASK_APP = ".\run.py"
$
$ # Set up the DEBUG environment
$ # (Unix/Mac) export FLASK_ENV=development
$ # (Windows) set FLASK_ENV=development
$ # (Powershell) $env:FLASK_ENV = "development"
$
$ # Start the application (development mode)
$ # --host=0.0.0.0 - expose the app on all network interfaces (default 127.0.0.1)
$ # --port=5000    - specify the app port (default 5000)  
$ flask run --host=0.0.0.0 --port=5000
$
$ # Access the dashboard in browser: http://127.0.0.1:5000/
```


<br />

## Credits & Links

- [Flask Framework](https://www.palletsprojects.com/p/flask/) - The offcial website
- [Boilerplate Code](https://appseed.us/boilerplate-code) - Index provided by **AppSeed**

<br />

---


