# External Code

## Systre (Olaf Delgado-Friedrichs)

Systre is a Java code written by Olaf Delagado Friedrichs. It is available as part of gavrog available here as source:

https://github.com/odf/gavrog

The precompiled jar file of systre is available here:

https://github.com/odf/gavrog/releases

RS: From 18/0521 systre.py does not run via jython but directly via java. This has several advantages.

To run systre.py in molsys.utils you need to download the current Systre jar-file (Systre-19.6.0.jar) into this directory. You need a java RT liek OpenJDK. If molsys is located in ```~/sandbox/molsys``` put the 
jar file into /external and set CLASSPATH like this:
```
export CLASSPATH=~/sandbox/molsys/external/Systre.jar
```
NOTE: with java the systreCmd.py is not needed any more. I did not test it but also the CLASSPATH environment varibale should not be necessary.
Please report problems to rochus.schmid@rub.de if there are any issues with systre.py and the java option.

## systrekey.js


For this to work you need to have the javascript systrekey code from Olaf Delgado-Friedrichs installed.

You need to have a working node.js environment working (do not use the distro version .. they are often too old) from https://nodejs.org

then clone the systreKey repo from here https://github.com/odf/systreKey to the same directory where your molsys repo is located (e.g. ~/sandbox)
because systre_path is determined from the root path of molsys.
For the installation follow the instructions from the systreKey repos README 

```
    git clone https://github.com/odf/systreKey.git
    cd systreKey
    npm install
    npm run build
```

Do not forget to run "npm run build" every time you pulled updates from Olaf's repo.
