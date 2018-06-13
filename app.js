var express = require('express');
var PythonShell = require('python-shell');
var cors = require('cors');
var path = require('path');
var app = express();


app.use(cors())
app.use(express.static(__dirname));

app.get('/', function(req, res) {
    res.sendFile(path.join(__dirname + '/index.html'));
});
app.get('/predict', predict);

function predict (request, response) {
  console.log("Predict")
  var options = {
    mode: 'text',
    pythonOptions: ['-u'], // get print results in real-time
    args: [request.query.dt, request.query.weather, request.query.temp, request.query.humidity, request.query.windspeed]
  };
  var results;
  PythonShell.run('test.py', options, function (err, res) {
    
    //res is an array consisting of messages collected during execution
    results = res[0].replace('\r','');
    console.log(res);
    response.json({"prediction" : results});
  });
}


app.listen(3000, function () {
  console.log('Server running on port 3000');
})