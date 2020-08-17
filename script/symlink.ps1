#cmd /c mklink /d rl ..\rl
#cmd /c mklink /d tmp ..\tmp
#
#New-Item -ItemType SymbolicLink -Path rl -Target ../rl
#New-Item -ItemType SymbolicLink -Path tmp -Target ../tmp
ln -s ../rl rl
ln -s ../tmp tmp
