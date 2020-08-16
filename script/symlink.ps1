#cmd /c mklink /d june60 ..\june60
#cmd /c mklink /d tmp ..\tmp
New-Item -ItemType SymbolicLink -Path june60 -Target ../june60
New-Item -ItemType SymbolicLink -Path tmp -Target ../tmp
