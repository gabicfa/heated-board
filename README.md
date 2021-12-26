# Forinsper
Flask / Python forum by Insper students for Insper students

### Dependencies:

* Python 3.4
* Flask (python-flask)
* SQLite3
* BCrypt
* Flask.ext.wtf

### Setup

#### Install dependencies:

* Ubuntu:
```
sudo apt-get install python-flask sqlite3 python-bcrypt python-flaskext.wtf
```

* Arch Linux:
```
yaourt -S python-flask python-flask-wtf python-flask-bcrypt sqlite3
```

* pip (you may need to run the command as root, also, you'll need to install sqlite3 from elsewhere):
```
pip install Flask flask-wtf bcrypt
```

#### Starting the server

* If there isn't already a DB, build one by using
```
sqlite3 flask-forum.db < schema.sql
```

* Start the server with
```
python app.py
```

* The server should be running on localhost:5000