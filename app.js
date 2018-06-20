var express = require('express');
var PythonShell = require('python-shell');
var cors = require('cors');
var path = require('path');
var app = express();
var fs = require('fs');


app.use(cors())
app.use(express.static(__dirname));
app.use(express.json());


app.get('/', function(req, res) {
	res.sendFile(path.join(__dirname + '/index.html'));
});
app.post('/predict', predict);

function predict (request, response) {
  	console.log("Predict")
  	json = request.body
  	filename = "input.csv"
  	output = "timestamp,weathersit,temp,hum,windspeed\n"

  	for(var i in json.list){
  		//console.log("hello, " +json.list[i])
  		output += json.list[i].dt + "," + json.list[i].weather[0].main +","+ json.list[i].main.temp+","+json.list[i].main.humidity+","+json.list[i].wind.speed+"\n"
  	}


	fs.writeFile("input.csv", output , function(err) {
		if(err) {
			return console.log(err);
		}

		console.log("The file was saved!");
	});
	var options = {
		mode: 'text',
		pythonOptions: ['-u'], // get print results in real-time
		args: ["input.csv"]
  	};
  	var results;
  	PythonShell.run('test.py', options, function (err, res) {
		if (err) throw err;
		//res is an array consisting of messages collected during execution
		for (var i = res.length - 1; i >= 0; i--) {
			res[i] = res[i].replace('\r','');
		}
		console.log(res);
		response.json({"prediction" : res});
  	});
}


app.listen(3000, function () {
  console.log('Server running on port 3000');
})