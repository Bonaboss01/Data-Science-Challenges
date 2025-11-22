from flask import url_for
@app.route('/admin')
def admin():
    return redirect(url_for('login'))
@app.route('/login')
def login():
    return "<Login Page>"


# Handing different HTTP request types

@app.route('/data', methods=['GET', 'POST'])
def data():
    if request.method == 'POST':
        # process POST request
    if request.method == 'GET':
        # process GET request


<!-- For POST -->
<form method="POST" action="/data">
    <!-- Your input fields here -->
    <input type="submit" value="Submit">
</form>
<!-- For GET -->
<a href="/data">Fetch data</a>
