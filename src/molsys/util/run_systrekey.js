// RS 2019 .. revised code to run systrekey with mapping using json for exchange
//            of data with python

// parse input
var lqg = JSON.parse(process.argv[2]);
var path = process.argv[3];

// load package, assuming that systreKey lives in the same dir as molsys (usually the git repository dir)
const sk = require(path + '/dist/systreKey').systreKeyWithMapping;

// get key and write it to stdout
let out = sk(lqg);
console.log(JSON.stringify(out));



