??
??
:
Add
x"T
y"T
z"T"
Ttype:
2	
B
AddV2
x"T
y"T
z"T"
Ttype:
2	??
B
AssignVariableOp
resource
value"dtype"
dtypetype?
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
?
Conv2D

input"T
filter"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

?
Conv2DBackpropInput
input_sizes
filter"T
out_backprop"T
output"T"
Ttype:	
2"
strides	list(int)"
use_cudnn_on_gpubool(",
paddingstring:
SAMEVALIDEXPLICIT""
explicit_paddings	list(int)
 "-
data_formatstringNHWC:
NHWCNCHW" 
	dilations	list(int)

,
Exp
x"T
y"T"
Ttype:

2
B
GreaterEqual
x"T
y"T
z
"
Ttype:
2	
.
Identity

input"T
output"T"	
Ttype
,
Log
x"T
y"T"
Ttype:

2
.
Log1p
x"T
y"T"
Ttype:

2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
?
Mean

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(?
=
Mul
x"T
y"T
z"T"
Ttype:
2	?
0
Neg
x"T
y"T"
Ttype:
2
	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
?
PartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
6
Pow
x"T
y"T
z"T"
Ttype:

2	
?
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	?
@
ReadVariableOp
resource
value"dtype"
dtypetype?
E
Relu
features"T
activations"T"
Ttype:
2	
[
Reshape
tensor"T
shape"Tshape
output"T"	
Ttype"
Tshapetype0:
2	
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0?
?
Select
	condition

t"T
e"T
output"T"	
Ttype
P
Shape

input"T
output"out_type"	
Ttype"
out_typetype0:
2	
H
ShardedFilename
basename	
shard

num_shards
filename
?
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring ?
@
StaticRegexFullMatch	
input

output
"
patternstring
?
StridedSlice

input"T
begin"Index
end"Index
strides"Index
output"T"	
Ttype"
Indextype:
2	"

begin_maskint "
end_maskint "
ellipsis_maskint "
new_axis_maskint "
shrink_axis_maskint 
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
;
Sub
x"T
y"T
z"T"
Ttype:
2	
?
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
?
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 ?
&
	ZerosLike
x"T
y"T"	
Ttype"serve*2.4.12v2.4.0-49-g85c8b2a817f8??
f
	Adam/iterVarHandleOp*
_output_shapes
: *
dtype0	*
shape: *
shared_name	Adam/iter
_
Adam/iter/Read/ReadVariableOpReadVariableOp	Adam/iter*
_output_shapes
: *
dtype0	
j
Adam/beta_1VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_1
c
Adam/beta_1/Read/ReadVariableOpReadVariableOpAdam/beta_1*
_output_shapes
: *
dtype0
j
Adam/beta_2VarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nameAdam/beta_2
c
Adam/beta_2/Read/ReadVariableOpReadVariableOpAdam/beta_2*
_output_shapes
: *
dtype0
h

Adam/decayVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_name
Adam/decay
a
Adam/decay/Read/ReadVariableOpReadVariableOp
Adam/decay*
_output_shapes
: *
dtype0
x
Adam/learning_rateVarHandleOp*
_output_shapes
: *
dtype0*
shape: *#
shared_nameAdam/learning_rate
q
&Adam/learning_rate/Read/ReadVariableOpReadVariableOpAdam/learning_rate*
_output_shapes
: *
dtype0
?
&var_auto_encoder/encoder/conv2d/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *7
shared_name(&var_auto_encoder/encoder/conv2d/kernel
?
:var_auto_encoder/encoder/conv2d/kernel/Read/ReadVariableOpReadVariableOp&var_auto_encoder/encoder/conv2d/kernel*&
_output_shapes
: *
dtype0
?
$var_auto_encoder/encoder/conv2d/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *5
shared_name&$var_auto_encoder/encoder/conv2d/bias
?
8var_auto_encoder/encoder/conv2d/bias/Read/ReadVariableOpReadVariableOp$var_auto_encoder/encoder/conv2d/bias*
_output_shapes
: *
dtype0
?
(var_auto_encoder/encoder/conv2d_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*9
shared_name*(var_auto_encoder/encoder/conv2d_1/kernel
?
<var_auto_encoder/encoder/conv2d_1/kernel/Read/ReadVariableOpReadVariableOp(var_auto_encoder/encoder/conv2d_1/kernel*&
_output_shapes
: @*
dtype0
?
&var_auto_encoder/encoder/conv2d_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*7
shared_name(&var_auto_encoder/encoder/conv2d_1/bias
?
:var_auto_encoder/encoder/conv2d_1/bias/Read/ReadVariableOpReadVariableOp&var_auto_encoder/encoder/conv2d_1/bias*
_output_shapes
:@*
dtype0
?
%var_auto_encoder/encoder/dense/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*6
shared_name'%var_auto_encoder/encoder/dense/kernel
?
9var_auto_encoder/encoder/dense/kernel/Read/ReadVariableOpReadVariableOp%var_auto_encoder/encoder/dense/kernel*
_output_shapes
:	?*
dtype0
?
#var_auto_encoder/encoder/dense/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*4
shared_name%#var_auto_encoder/encoder/dense/bias
?
7var_auto_encoder/encoder/dense/bias/Read/ReadVariableOpReadVariableOp#var_auto_encoder/encoder/dense/bias*
_output_shapes
:*
dtype0
?
'var_auto_encoder/encoder/dense_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*8
shared_name)'var_auto_encoder/encoder/dense_1/kernel
?
;var_auto_encoder/encoder/dense_1/kernel/Read/ReadVariableOpReadVariableOp'var_auto_encoder/encoder/dense_1/kernel*
_output_shapes
:	?*
dtype0
?
%var_auto_encoder/encoder/dense_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*6
shared_name'%var_auto_encoder/encoder/dense_1/bias
?
9var_auto_encoder/encoder/dense_1/bias/Read/ReadVariableOpReadVariableOp%var_auto_encoder/encoder/dense_1/bias*
_output_shapes
:*
dtype0
?
'var_auto_encoder/decoder/dense_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*8
shared_name)'var_auto_encoder/decoder/dense_2/kernel
?
;var_auto_encoder/decoder/dense_2/kernel/Read/ReadVariableOpReadVariableOp'var_auto_encoder/decoder/dense_2/kernel*
_output_shapes
:	?*
dtype0
?
%var_auto_encoder/decoder/dense_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*6
shared_name'%var_auto_encoder/decoder/dense_2/bias
?
9var_auto_encoder/decoder/dense_2/bias/Read/ReadVariableOpReadVariableOp%var_auto_encoder/decoder/dense_2/bias*
_output_shapes	
:?*
dtype0
?
0var_auto_encoder/decoder/conv2d_transpose/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *A
shared_name20var_auto_encoder/decoder/conv2d_transpose/kernel
?
Dvar_auto_encoder/decoder/conv2d_transpose/kernel/Read/ReadVariableOpReadVariableOp0var_auto_encoder/decoder/conv2d_transpose/kernel*&
_output_shapes
:@ *
dtype0
?
.var_auto_encoder/decoder/conv2d_transpose/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*?
shared_name0.var_auto_encoder/decoder/conv2d_transpose/bias
?
Bvar_auto_encoder/decoder/conv2d_transpose/bias/Read/ReadVariableOpReadVariableOp.var_auto_encoder/decoder/conv2d_transpose/bias*
_output_shapes
:@*
dtype0
?
2var_auto_encoder/decoder/conv2d_transpose_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*C
shared_name42var_auto_encoder/decoder/conv2d_transpose_1/kernel
?
Fvar_auto_encoder/decoder/conv2d_transpose_1/kernel/Read/ReadVariableOpReadVariableOp2var_auto_encoder/decoder/conv2d_transpose_1/kernel*&
_output_shapes
: @*
dtype0
?
0var_auto_encoder/decoder/conv2d_transpose_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape: *A
shared_name20var_auto_encoder/decoder/conv2d_transpose_1/bias
?
Dvar_auto_encoder/decoder/conv2d_transpose_1/bias/Read/ReadVariableOpReadVariableOp0var_auto_encoder/decoder/conv2d_transpose_1/bias*
_output_shapes
: *
dtype0
?
2var_auto_encoder/decoder/conv2d_transpose_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape: *C
shared_name42var_auto_encoder/decoder/conv2d_transpose_2/kernel
?
Fvar_auto_encoder/decoder/conv2d_transpose_2/kernel/Read/ReadVariableOpReadVariableOp2var_auto_encoder/decoder/conv2d_transpose_2/kernel*&
_output_shapes
: *
dtype0
?
0var_auto_encoder/decoder/conv2d_transpose_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*A
shared_name20var_auto_encoder/decoder/conv2d_transpose_2/bias
?
Dvar_auto_encoder/decoder/conv2d_transpose_2/bias/Read/ReadVariableOpReadVariableOp0var_auto_encoder/decoder/conv2d_transpose_2/bias*
_output_shapes
:*
dtype0
^
totalVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_nametotal
W
total/Read/ReadVariableOpReadVariableOptotal*
_output_shapes
: *
dtype0
^
countVarHandleOp*
_output_shapes
: *
dtype0*
shape: *
shared_namecount
W
count/Read/ReadVariableOpReadVariableOpcount*
_output_shapes
: *
dtype0
?
-Adam/var_auto_encoder/encoder/conv2d/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/var_auto_encoder/encoder/conv2d/kernel/m
?
AAdam/var_auto_encoder/encoder/conv2d/kernel/m/Read/ReadVariableOpReadVariableOp-Adam/var_auto_encoder/encoder/conv2d/kernel/m*&
_output_shapes
: *
dtype0
?
+Adam/var_auto_encoder/encoder/conv2d/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/var_auto_encoder/encoder/conv2d/bias/m
?
?Adam/var_auto_encoder/encoder/conv2d/bias/m/Read/ReadVariableOpReadVariableOp+Adam/var_auto_encoder/encoder/conv2d/bias/m*
_output_shapes
: *
dtype0
?
/Adam/var_auto_encoder/encoder/conv2d_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*@
shared_name1/Adam/var_auto_encoder/encoder/conv2d_1/kernel/m
?
CAdam/var_auto_encoder/encoder/conv2d_1/kernel/m/Read/ReadVariableOpReadVariableOp/Adam/var_auto_encoder/encoder/conv2d_1/kernel/m*&
_output_shapes
: @*
dtype0
?
-Adam/var_auto_encoder/encoder/conv2d_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-Adam/var_auto_encoder/encoder/conv2d_1/bias/m
?
AAdam/var_auto_encoder/encoder/conv2d_1/bias/m/Read/ReadVariableOpReadVariableOp-Adam/var_auto_encoder/encoder/conv2d_1/bias/m*
_output_shapes
:@*
dtype0
?
,Adam/var_auto_encoder/encoder/dense/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*=
shared_name.,Adam/var_auto_encoder/encoder/dense/kernel/m
?
@Adam/var_auto_encoder/encoder/dense/kernel/m/Read/ReadVariableOpReadVariableOp,Adam/var_auto_encoder/encoder/dense/kernel/m*
_output_shapes
:	?*
dtype0
?
*Adam/var_auto_encoder/encoder/dense/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/var_auto_encoder/encoder/dense/bias/m
?
>Adam/var_auto_encoder/encoder/dense/bias/m/Read/ReadVariableOpReadVariableOp*Adam/var_auto_encoder/encoder/dense/bias/m*
_output_shapes
:*
dtype0
?
.Adam/var_auto_encoder/encoder/dense_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.Adam/var_auto_encoder/encoder/dense_1/kernel/m
?
BAdam/var_auto_encoder/encoder/dense_1/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/var_auto_encoder/encoder/dense_1/kernel/m*
_output_shapes
:	?*
dtype0
?
,Adam/var_auto_encoder/encoder/dense_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/var_auto_encoder/encoder/dense_1/bias/m
?
@Adam/var_auto_encoder/encoder/dense_1/bias/m/Read/ReadVariableOpReadVariableOp,Adam/var_auto_encoder/encoder/dense_1/bias/m*
_output_shapes
:*
dtype0
?
.Adam/var_auto_encoder/decoder/dense_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.Adam/var_auto_encoder/decoder/dense_2/kernel/m
?
BAdam/var_auto_encoder/decoder/dense_2/kernel/m/Read/ReadVariableOpReadVariableOp.Adam/var_auto_encoder/decoder/dense_2/kernel/m*
_output_shapes
:	?*
dtype0
?
,Adam/var_auto_encoder/decoder/dense_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/var_auto_encoder/decoder/dense_2/bias/m
?
@Adam/var_auto_encoder/decoder/dense_2/bias/m/Read/ReadVariableOpReadVariableOp,Adam/var_auto_encoder/decoder/dense_2/bias/m*
_output_shapes	
:?*
dtype0
?
7Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *H
shared_name97Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/m
?
KAdam/var_auto_encoder/decoder/conv2d_transpose/kernel/m/Read/ReadVariableOpReadVariableOp7Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/m*&
_output_shapes
:@ *
dtype0
?
5Adam/var_auto_encoder/decoder/conv2d_transpose/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/var_auto_encoder/decoder/conv2d_transpose/bias/m
?
IAdam/var_auto_encoder/decoder/conv2d_transpose/bias/m/Read/ReadVariableOpReadVariableOp5Adam/var_auto_encoder/decoder/conv2d_transpose/bias/m*
_output_shapes
:@*
dtype0
?
9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*J
shared_name;9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/m
?
MAdam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/m/Read/ReadVariableOpReadVariableOp9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/m*&
_output_shapes
: @*
dtype0
?
7Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/m
?
KAdam/var_auto_encoder/decoder/conv2d_transpose_1/bias/m/Read/ReadVariableOpReadVariableOp7Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/m*
_output_shapes
: *
dtype0
?
9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/mVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/m
?
MAdam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/m/Read/ReadVariableOpReadVariableOp9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/m*&
_output_shapes
: *
dtype0
?
7Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/mVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/m
?
KAdam/var_auto_encoder/decoder/conv2d_transpose_2/bias/m/Read/ReadVariableOpReadVariableOp7Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/m*
_output_shapes
:*
dtype0
?
-Adam/var_auto_encoder/encoder/conv2d/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *>
shared_name/-Adam/var_auto_encoder/encoder/conv2d/kernel/v
?
AAdam/var_auto_encoder/encoder/conv2d/kernel/v/Read/ReadVariableOpReadVariableOp-Adam/var_auto_encoder/encoder/conv2d/kernel/v*&
_output_shapes
: *
dtype0
?
+Adam/var_auto_encoder/encoder/conv2d/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *<
shared_name-+Adam/var_auto_encoder/encoder/conv2d/bias/v
?
?Adam/var_auto_encoder/encoder/conv2d/bias/v/Read/ReadVariableOpReadVariableOp+Adam/var_auto_encoder/encoder/conv2d/bias/v*
_output_shapes
: *
dtype0
?
/Adam/var_auto_encoder/encoder/conv2d_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*@
shared_name1/Adam/var_auto_encoder/encoder/conv2d_1/kernel/v
?
CAdam/var_auto_encoder/encoder/conv2d_1/kernel/v/Read/ReadVariableOpReadVariableOp/Adam/var_auto_encoder/encoder/conv2d_1/kernel/v*&
_output_shapes
: @*
dtype0
?
-Adam/var_auto_encoder/encoder/conv2d_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*>
shared_name/-Adam/var_auto_encoder/encoder/conv2d_1/bias/v
?
AAdam/var_auto_encoder/encoder/conv2d_1/bias/v/Read/ReadVariableOpReadVariableOp-Adam/var_auto_encoder/encoder/conv2d_1/bias/v*
_output_shapes
:@*
dtype0
?
,Adam/var_auto_encoder/encoder/dense/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*=
shared_name.,Adam/var_auto_encoder/encoder/dense/kernel/v
?
@Adam/var_auto_encoder/encoder/dense/kernel/v/Read/ReadVariableOpReadVariableOp,Adam/var_auto_encoder/encoder/dense/kernel/v*
_output_shapes
:	?*
dtype0
?
*Adam/var_auto_encoder/encoder/dense/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*;
shared_name,*Adam/var_auto_encoder/encoder/dense/bias/v
?
>Adam/var_auto_encoder/encoder/dense/bias/v/Read/ReadVariableOpReadVariableOp*Adam/var_auto_encoder/encoder/dense/bias/v*
_output_shapes
:*
dtype0
?
.Adam/var_auto_encoder/encoder/dense_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.Adam/var_auto_encoder/encoder/dense_1/kernel/v
?
BAdam/var_auto_encoder/encoder/dense_1/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/var_auto_encoder/encoder/dense_1/kernel/v*
_output_shapes
:	?*
dtype0
?
,Adam/var_auto_encoder/encoder/dense_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*=
shared_name.,Adam/var_auto_encoder/encoder/dense_1/bias/v
?
@Adam/var_auto_encoder/encoder/dense_1/bias/v/Read/ReadVariableOpReadVariableOp,Adam/var_auto_encoder/encoder/dense_1/bias/v*
_output_shapes
:*
dtype0
?
.Adam/var_auto_encoder/decoder/dense_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:	?*?
shared_name0.Adam/var_auto_encoder/decoder/dense_2/kernel/v
?
BAdam/var_auto_encoder/decoder/dense_2/kernel/v/Read/ReadVariableOpReadVariableOp.Adam/var_auto_encoder/decoder/dense_2/kernel/v*
_output_shapes
:	?*
dtype0
?
,Adam/var_auto_encoder/decoder/dense_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:?*=
shared_name.,Adam/var_auto_encoder/decoder/dense_2/bias/v
?
@Adam/var_auto_encoder/decoder/dense_2/bias/v/Read/ReadVariableOpReadVariableOp,Adam/var_auto_encoder/decoder/dense_2/bias/v*
_output_shapes	
:?*
dtype0
?
7Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@ *H
shared_name97Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/v
?
KAdam/var_auto_encoder/decoder/conv2d_transpose/kernel/v/Read/ReadVariableOpReadVariableOp7Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/v*&
_output_shapes
:@ *
dtype0
?
5Adam/var_auto_encoder/decoder/conv2d_transpose/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:@*F
shared_name75Adam/var_auto_encoder/decoder/conv2d_transpose/bias/v
?
IAdam/var_auto_encoder/decoder/conv2d_transpose/bias/v/Read/ReadVariableOpReadVariableOp5Adam/var_auto_encoder/decoder/conv2d_transpose/bias/v*
_output_shapes
:@*
dtype0
?
9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: @*J
shared_name;9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/v
?
MAdam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/v/Read/ReadVariableOpReadVariableOp9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/v*&
_output_shapes
: @*
dtype0
?
7Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *H
shared_name97Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/v
?
KAdam/var_auto_encoder/decoder/conv2d_transpose_1/bias/v/Read/ReadVariableOpReadVariableOp7Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/v*
_output_shapes
: *
dtype0
?
9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/vVarHandleOp*
_output_shapes
: *
dtype0*
shape: *J
shared_name;9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/v
?
MAdam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/v/Read/ReadVariableOpReadVariableOp9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/v*&
_output_shapes
: *
dtype0
?
7Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/vVarHandleOp*
_output_shapes
: *
dtype0*
shape:*H
shared_name97Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/v
?
KAdam/var_auto_encoder/decoder/conv2d_transpose_2/bias/v/Read/ReadVariableOpReadVariableOp7Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/v*
_output_shapes
:*
dtype0

NoOpNoOp
?d
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*?c
value?cB?c B?c
?
encoder
decoder
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
?
	
conv1
	conv2
flatten
dense3_1
dense3_2
sampling
trainable_variables
regularization_losses
	variables
	keras_api
?

dense1
reshape
deconv1
deconv2
out
trainable_variables
regularization_losses
	variables
	keras_api
?
iter

beta_1

beta_2
	 decay
!learning_rate"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?
 
v
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
 
v
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115
?

2layers
3non_trainable_variables
trainable_variables
4layer_regularization_losses
regularization_losses
5metrics
6layer_metrics
	variables
 
h

"kernel
#bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
h

$kernel
%bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
R
?trainable_variables
@regularization_losses
A	variables
B	keras_api
h

&kernel
'bias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
h

(kernel
)bias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
R
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
8
"0
#1
$2
%3
&4
'5
(6
)7
 
8
"0
#1
$2
%3
&4
'5
(6
)7
?

Olayers
Pnon_trainable_variables
trainable_variables
Qlayer_regularization_losses
regularization_losses
Rmetrics
Slayer_metrics
	variables
h

*kernel
+bias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
R
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
h

,kernel
-bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
h

.kernel
/bias
`trainable_variables
aregularization_losses
b	variables
c	keras_api
h

0kernel
1bias
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
8
*0
+1
,2
-3
.4
/5
06
17
 
8
*0
+1
,2
-3
.4
/5
06
17
?

hlayers
inon_trainable_variables
trainable_variables
jlayer_regularization_losses
regularization_losses
kmetrics
llayer_metrics
	variables
HF
VARIABLE_VALUE	Adam/iter)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_1+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUE
LJ
VARIABLE_VALUEAdam/beta_2+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUE
JH
VARIABLE_VALUE
Adam/decay*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUE
ZX
VARIABLE_VALUEAdam/learning_rate2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE&var_auto_encoder/encoder/conv2d/kernel0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUE
jh
VARIABLE_VALUE$var_auto_encoder/encoder/conv2d/bias0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUE
nl
VARIABLE_VALUE(var_auto_encoder/encoder/conv2d_1/kernel0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUE
lj
VARIABLE_VALUE&var_auto_encoder/encoder/conv2d_1/bias0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%var_auto_encoder/encoder/dense/kernel0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUE
ig
VARIABLE_VALUE#var_auto_encoder/encoder/dense/bias0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'var_auto_encoder/encoder/dense_1/kernel0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%var_auto_encoder/encoder/dense_1/bias0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUE
mk
VARIABLE_VALUE'var_auto_encoder/decoder/dense_2/kernel0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUE
ki
VARIABLE_VALUE%var_auto_encoder/decoder/dense_2/bias0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0var_auto_encoder/decoder/conv2d_transpose/kernel1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUE
us
VARIABLE_VALUE.var_auto_encoder/decoder/conv2d_transpose/bias1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2var_auto_encoder/decoder/conv2d_transpose_1/kernel1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0var_auto_encoder/decoder/conv2d_transpose_1/bias1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUE
yw
VARIABLE_VALUE2var_auto_encoder/decoder/conv2d_transpose_2/kernel1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUE
wu
VARIABLE_VALUE0var_auto_encoder/decoder/conv2d_transpose_2/bias1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUE

0
1
 
 

m0
 

"0
#1
 

"0
#1
?

nlayers
onon_trainable_variables
7trainable_variables
player_regularization_losses
8regularization_losses
qmetrics
rlayer_metrics
9	variables

$0
%1
 

$0
%1
?

slayers
tnon_trainable_variables
;trainable_variables
ulayer_regularization_losses
<regularization_losses
vmetrics
wlayer_metrics
=	variables
 
 
 
?

xlayers
ynon_trainable_variables
?trainable_variables
zlayer_regularization_losses
@regularization_losses
{metrics
|layer_metrics
A	variables

&0
'1
 

&0
'1
?

}layers
~non_trainable_variables
Ctrainable_variables
layer_regularization_losses
Dregularization_losses
?metrics
?layer_metrics
E	variables

(0
)1
 

(0
)1
?
?layers
?non_trainable_variables
Gtrainable_variables
 ?layer_regularization_losses
Hregularization_losses
?metrics
?layer_metrics
I	variables
 
 
 
?
?layers
?non_trainable_variables
Ktrainable_variables
 ?layer_regularization_losses
Lregularization_losses
?metrics
?layer_metrics
M	variables
*

0
1
2
3
4
5
 
 
 
 

*0
+1
 

*0
+1
?
?layers
?non_trainable_variables
Ttrainable_variables
 ?layer_regularization_losses
Uregularization_losses
?metrics
?layer_metrics
V	variables
 
 
 
?
?layers
?non_trainable_variables
Xtrainable_variables
 ?layer_regularization_losses
Yregularization_losses
?metrics
?layer_metrics
Z	variables

,0
-1
 

,0
-1
?
?layers
?non_trainable_variables
\trainable_variables
 ?layer_regularization_losses
]regularization_losses
?metrics
?layer_metrics
^	variables

.0
/1
 

.0
/1
?
?layers
?non_trainable_variables
`trainable_variables
 ?layer_regularization_losses
aregularization_losses
?metrics
?layer_metrics
b	variables

00
11
 

00
11
?
?layers
?non_trainable_variables
dtrainable_variables
 ?layer_regularization_losses
eregularization_losses
?metrics
?layer_metrics
f	variables
#
0
1
2
3
4
 
 
 
 
8

?total

?count
?	variables
?	keras_api
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
OM
VARIABLE_VALUEtotal4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUE
OM
VARIABLE_VALUEcount4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUE

?0
?1

?	variables
??
VARIABLE_VALUE-Adam/var_auto_encoder/encoder/conv2d/kernel/mLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/var_auto_encoder/encoder/conv2d/bias/mLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/var_auto_encoder/encoder/conv2d_1/kernel/mLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/var_auto_encoder/encoder/conv2d_1/bias/mLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/var_auto_encoder/encoder/dense/kernel/mLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/var_auto_encoder/encoder/dense/bias/mLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/var_auto_encoder/encoder/dense_1/kernel/mLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/var_auto_encoder/encoder/dense_1/bias/mLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/var_auto_encoder/decoder/dense_2/kernel/mLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/var_auto_encoder/decoder/dense_2/bias/mLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/mMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/var_auto_encoder/decoder/conv2d_transpose/bias/mMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/mMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/mMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/mMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/mMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/var_auto_encoder/encoder/conv2d/kernel/vLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE+Adam/var_auto_encoder/encoder/conv2d/bias/vLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE/Adam/var_auto_encoder/encoder/conv2d_1/kernel/vLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE-Adam/var_auto_encoder/encoder/conv2d_1/bias/vLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/var_auto_encoder/encoder/dense/kernel/vLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE*Adam/var_auto_encoder/encoder/dense/bias/vLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/var_auto_encoder/encoder/dense_1/kernel/vLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/var_auto_encoder/encoder/dense_1/bias/vLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE.Adam/var_auto_encoder/decoder/dense_2/kernel/vLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE,Adam/var_auto_encoder/decoder/dense_2/bias/vLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/vMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE5Adam/var_auto_encoder/decoder/conv2d_transpose/bias/vMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/vMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/vMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/vMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
??
VARIABLE_VALUE7Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/vMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUE
?
serving_default_input_1Placeholder*/
_output_shapes
:?????????*
dtype0*$
shape:?????????
?
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1&var_auto_encoder/encoder/conv2d/kernel$var_auto_encoder/encoder/conv2d/bias(var_auto_encoder/encoder/conv2d_1/kernel&var_auto_encoder/encoder/conv2d_1/bias%var_auto_encoder/encoder/dense/kernel#var_auto_encoder/encoder/dense/bias'var_auto_encoder/encoder/dense_1/kernel%var_auto_encoder/encoder/dense_1/bias'var_auto_encoder/decoder/dense_2/kernel%var_auto_encoder/decoder/dense_2/bias0var_auto_encoder/decoder/conv2d_transpose/kernel.var_auto_encoder/decoder/conv2d_transpose/bias2var_auto_encoder/decoder/conv2d_transpose_1/kernel0var_auto_encoder/decoder/conv2d_transpose_1/bias2var_auto_encoder/decoder/conv2d_transpose_2/kernel0var_auto_encoder/decoder/conv2d_transpose_2/bias*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *,
f'R%
#__inference_signature_wrapper_60056
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
?
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filenameAdam/iter/Read/ReadVariableOpAdam/beta_1/Read/ReadVariableOpAdam/beta_2/Read/ReadVariableOpAdam/decay/Read/ReadVariableOp&Adam/learning_rate/Read/ReadVariableOp:var_auto_encoder/encoder/conv2d/kernel/Read/ReadVariableOp8var_auto_encoder/encoder/conv2d/bias/Read/ReadVariableOp<var_auto_encoder/encoder/conv2d_1/kernel/Read/ReadVariableOp:var_auto_encoder/encoder/conv2d_1/bias/Read/ReadVariableOp9var_auto_encoder/encoder/dense/kernel/Read/ReadVariableOp7var_auto_encoder/encoder/dense/bias/Read/ReadVariableOp;var_auto_encoder/encoder/dense_1/kernel/Read/ReadVariableOp9var_auto_encoder/encoder/dense_1/bias/Read/ReadVariableOp;var_auto_encoder/decoder/dense_2/kernel/Read/ReadVariableOp9var_auto_encoder/decoder/dense_2/bias/Read/ReadVariableOpDvar_auto_encoder/decoder/conv2d_transpose/kernel/Read/ReadVariableOpBvar_auto_encoder/decoder/conv2d_transpose/bias/Read/ReadVariableOpFvar_auto_encoder/decoder/conv2d_transpose_1/kernel/Read/ReadVariableOpDvar_auto_encoder/decoder/conv2d_transpose_1/bias/Read/ReadVariableOpFvar_auto_encoder/decoder/conv2d_transpose_2/kernel/Read/ReadVariableOpDvar_auto_encoder/decoder/conv2d_transpose_2/bias/Read/ReadVariableOptotal/Read/ReadVariableOpcount/Read/ReadVariableOpAAdam/var_auto_encoder/encoder/conv2d/kernel/m/Read/ReadVariableOp?Adam/var_auto_encoder/encoder/conv2d/bias/m/Read/ReadVariableOpCAdam/var_auto_encoder/encoder/conv2d_1/kernel/m/Read/ReadVariableOpAAdam/var_auto_encoder/encoder/conv2d_1/bias/m/Read/ReadVariableOp@Adam/var_auto_encoder/encoder/dense/kernel/m/Read/ReadVariableOp>Adam/var_auto_encoder/encoder/dense/bias/m/Read/ReadVariableOpBAdam/var_auto_encoder/encoder/dense_1/kernel/m/Read/ReadVariableOp@Adam/var_auto_encoder/encoder/dense_1/bias/m/Read/ReadVariableOpBAdam/var_auto_encoder/decoder/dense_2/kernel/m/Read/ReadVariableOp@Adam/var_auto_encoder/decoder/dense_2/bias/m/Read/ReadVariableOpKAdam/var_auto_encoder/decoder/conv2d_transpose/kernel/m/Read/ReadVariableOpIAdam/var_auto_encoder/decoder/conv2d_transpose/bias/m/Read/ReadVariableOpMAdam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/m/Read/ReadVariableOpKAdam/var_auto_encoder/decoder/conv2d_transpose_1/bias/m/Read/ReadVariableOpMAdam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/m/Read/ReadVariableOpKAdam/var_auto_encoder/decoder/conv2d_transpose_2/bias/m/Read/ReadVariableOpAAdam/var_auto_encoder/encoder/conv2d/kernel/v/Read/ReadVariableOp?Adam/var_auto_encoder/encoder/conv2d/bias/v/Read/ReadVariableOpCAdam/var_auto_encoder/encoder/conv2d_1/kernel/v/Read/ReadVariableOpAAdam/var_auto_encoder/encoder/conv2d_1/bias/v/Read/ReadVariableOp@Adam/var_auto_encoder/encoder/dense/kernel/v/Read/ReadVariableOp>Adam/var_auto_encoder/encoder/dense/bias/v/Read/ReadVariableOpBAdam/var_auto_encoder/encoder/dense_1/kernel/v/Read/ReadVariableOp@Adam/var_auto_encoder/encoder/dense_1/bias/v/Read/ReadVariableOpBAdam/var_auto_encoder/decoder/dense_2/kernel/v/Read/ReadVariableOp@Adam/var_auto_encoder/decoder/dense_2/bias/v/Read/ReadVariableOpKAdam/var_auto_encoder/decoder/conv2d_transpose/kernel/v/Read/ReadVariableOpIAdam/var_auto_encoder/decoder/conv2d_transpose/bias/v/Read/ReadVariableOpMAdam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/v/Read/ReadVariableOpKAdam/var_auto_encoder/decoder/conv2d_transpose_1/bias/v/Read/ReadVariableOpMAdam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/v/Read/ReadVariableOpKAdam/var_auto_encoder/decoder/conv2d_transpose_2/bias/v/Read/ReadVariableOpConst*D
Tin=
;29	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *'
f"R 
__inference__traced_save_60982
?
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filename	Adam/iterAdam/beta_1Adam/beta_2
Adam/decayAdam/learning_rate&var_auto_encoder/encoder/conv2d/kernel$var_auto_encoder/encoder/conv2d/bias(var_auto_encoder/encoder/conv2d_1/kernel&var_auto_encoder/encoder/conv2d_1/bias%var_auto_encoder/encoder/dense/kernel#var_auto_encoder/encoder/dense/bias'var_auto_encoder/encoder/dense_1/kernel%var_auto_encoder/encoder/dense_1/bias'var_auto_encoder/decoder/dense_2/kernel%var_auto_encoder/decoder/dense_2/bias0var_auto_encoder/decoder/conv2d_transpose/kernel.var_auto_encoder/decoder/conv2d_transpose/bias2var_auto_encoder/decoder/conv2d_transpose_1/kernel0var_auto_encoder/decoder/conv2d_transpose_1/bias2var_auto_encoder/decoder/conv2d_transpose_2/kernel0var_auto_encoder/decoder/conv2d_transpose_2/biastotalcount-Adam/var_auto_encoder/encoder/conv2d/kernel/m+Adam/var_auto_encoder/encoder/conv2d/bias/m/Adam/var_auto_encoder/encoder/conv2d_1/kernel/m-Adam/var_auto_encoder/encoder/conv2d_1/bias/m,Adam/var_auto_encoder/encoder/dense/kernel/m*Adam/var_auto_encoder/encoder/dense/bias/m.Adam/var_auto_encoder/encoder/dense_1/kernel/m,Adam/var_auto_encoder/encoder/dense_1/bias/m.Adam/var_auto_encoder/decoder/dense_2/kernel/m,Adam/var_auto_encoder/decoder/dense_2/bias/m7Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/m5Adam/var_auto_encoder/decoder/conv2d_transpose/bias/m9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/m7Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/m9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/m7Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/m-Adam/var_auto_encoder/encoder/conv2d/kernel/v+Adam/var_auto_encoder/encoder/conv2d/bias/v/Adam/var_auto_encoder/encoder/conv2d_1/kernel/v-Adam/var_auto_encoder/encoder/conv2d_1/bias/v,Adam/var_auto_encoder/encoder/dense/kernel/v*Adam/var_auto_encoder/encoder/dense/bias/v.Adam/var_auto_encoder/encoder/dense_1/kernel/v,Adam/var_auto_encoder/encoder/dense_1/bias/v.Adam/var_auto_encoder/decoder/dense_2/kernel/v,Adam/var_auto_encoder/decoder/dense_2/bias/v7Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/v5Adam/var_auto_encoder/decoder/conv2d_transpose/bias/v9Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/v7Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/v9Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/v7Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/v*C
Tin<
:28*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? **
f%R#
!__inference__traced_restore_61157??
?:
?
B__inference_encoder_layer_call_and_return_conditional_losses_60497

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity

identity_1

identity_2??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten/Const?
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddf
sampling/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean?
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sampling/random_normal/stddev?
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02-
+sampling/random_normal/RandomStandardNormal?
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
sampling/random_normal/mul?
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x?
sampling/mulMulsampling/mul/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling/Exp?
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling/mul_1?
sampling/addAddV2dense/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling/add?
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitysampling/add:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*N
_input_shapes=
;:?????????::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?u
?
B__inference_decoder_layer_call_and_return_conditional_losses_59666

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Reluh
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose/BiasAdd?
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose/Relu?
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/Relu?
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_2/BiasAdd?
IdentityIdentity#conv2d_transpose_2/BiasAdd:output:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
0__inference_conv2d_transpose_layer_call_fn_59252

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????@*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_592422
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
??
?&
!__inference__traced_restore_61157
file_prefix
assignvariableop_adam_iter"
assignvariableop_1_adam_beta_1"
assignvariableop_2_adam_beta_2!
assignvariableop_3_adam_decay)
%assignvariableop_4_adam_learning_rate=
9assignvariableop_5_var_auto_encoder_encoder_conv2d_kernel;
7assignvariableop_6_var_auto_encoder_encoder_conv2d_bias?
;assignvariableop_7_var_auto_encoder_encoder_conv2d_1_kernel=
9assignvariableop_8_var_auto_encoder_encoder_conv2d_1_bias<
8assignvariableop_9_var_auto_encoder_encoder_dense_kernel;
7assignvariableop_10_var_auto_encoder_encoder_dense_bias?
;assignvariableop_11_var_auto_encoder_encoder_dense_1_kernel=
9assignvariableop_12_var_auto_encoder_encoder_dense_1_bias?
;assignvariableop_13_var_auto_encoder_decoder_dense_2_kernel=
9assignvariableop_14_var_auto_encoder_decoder_dense_2_biasH
Dassignvariableop_15_var_auto_encoder_decoder_conv2d_transpose_kernelF
Bassignvariableop_16_var_auto_encoder_decoder_conv2d_transpose_biasJ
Fassignvariableop_17_var_auto_encoder_decoder_conv2d_transpose_1_kernelH
Dassignvariableop_18_var_auto_encoder_decoder_conv2d_transpose_1_biasJ
Fassignvariableop_19_var_auto_encoder_decoder_conv2d_transpose_2_kernelH
Dassignvariableop_20_var_auto_encoder_decoder_conv2d_transpose_2_bias
assignvariableop_21_total
assignvariableop_22_countE
Aassignvariableop_23_adam_var_auto_encoder_encoder_conv2d_kernel_mC
?assignvariableop_24_adam_var_auto_encoder_encoder_conv2d_bias_mG
Cassignvariableop_25_adam_var_auto_encoder_encoder_conv2d_1_kernel_mE
Aassignvariableop_26_adam_var_auto_encoder_encoder_conv2d_1_bias_mD
@assignvariableop_27_adam_var_auto_encoder_encoder_dense_kernel_mB
>assignvariableop_28_adam_var_auto_encoder_encoder_dense_bias_mF
Bassignvariableop_29_adam_var_auto_encoder_encoder_dense_1_kernel_mD
@assignvariableop_30_adam_var_auto_encoder_encoder_dense_1_bias_mF
Bassignvariableop_31_adam_var_auto_encoder_decoder_dense_2_kernel_mD
@assignvariableop_32_adam_var_auto_encoder_decoder_dense_2_bias_mO
Kassignvariableop_33_adam_var_auto_encoder_decoder_conv2d_transpose_kernel_mM
Iassignvariableop_34_adam_var_auto_encoder_decoder_conv2d_transpose_bias_mQ
Massignvariableop_35_adam_var_auto_encoder_decoder_conv2d_transpose_1_kernel_mO
Kassignvariableop_36_adam_var_auto_encoder_decoder_conv2d_transpose_1_bias_mQ
Massignvariableop_37_adam_var_auto_encoder_decoder_conv2d_transpose_2_kernel_mO
Kassignvariableop_38_adam_var_auto_encoder_decoder_conv2d_transpose_2_bias_mE
Aassignvariableop_39_adam_var_auto_encoder_encoder_conv2d_kernel_vC
?assignvariableop_40_adam_var_auto_encoder_encoder_conv2d_bias_vG
Cassignvariableop_41_adam_var_auto_encoder_encoder_conv2d_1_kernel_vE
Aassignvariableop_42_adam_var_auto_encoder_encoder_conv2d_1_bias_vD
@assignvariableop_43_adam_var_auto_encoder_encoder_dense_kernel_vB
>assignvariableop_44_adam_var_auto_encoder_encoder_dense_bias_vF
Bassignvariableop_45_adam_var_auto_encoder_encoder_dense_1_kernel_vD
@assignvariableop_46_adam_var_auto_encoder_encoder_dense_1_bias_vF
Bassignvariableop_47_adam_var_auto_encoder_decoder_dense_2_kernel_vD
@assignvariableop_48_adam_var_auto_encoder_decoder_dense_2_bias_vO
Kassignvariableop_49_adam_var_auto_encoder_decoder_conv2d_transpose_kernel_vM
Iassignvariableop_50_adam_var_auto_encoder_decoder_conv2d_transpose_bias_vQ
Massignvariableop_51_adam_var_auto_encoder_decoder_conv2d_transpose_1_kernel_vO
Kassignvariableop_52_adam_var_auto_encoder_decoder_conv2d_transpose_1_bias_vQ
Massignvariableop_53_adam_var_auto_encoder_decoder_conv2d_transpose_2_kernel_vO
Kassignvariableop_54_adam_var_auto_encoder_decoder_conv2d_transpose_2_bias_v
identity_56??AssignVariableOp?AssignVariableOp_1?AssignVariableOp_10?AssignVariableOp_11?AssignVariableOp_12?AssignVariableOp_13?AssignVariableOp_14?AssignVariableOp_15?AssignVariableOp_16?AssignVariableOp_17?AssignVariableOp_18?AssignVariableOp_19?AssignVariableOp_2?AssignVariableOp_20?AssignVariableOp_21?AssignVariableOp_22?AssignVariableOp_23?AssignVariableOp_24?AssignVariableOp_25?AssignVariableOp_26?AssignVariableOp_27?AssignVariableOp_28?AssignVariableOp_29?AssignVariableOp_3?AssignVariableOp_30?AssignVariableOp_31?AssignVariableOp_32?AssignVariableOp_33?AssignVariableOp_34?AssignVariableOp_35?AssignVariableOp_36?AssignVariableOp_37?AssignVariableOp_38?AssignVariableOp_39?AssignVariableOp_4?AssignVariableOp_40?AssignVariableOp_41?AssignVariableOp_42?AssignVariableOp_43?AssignVariableOp_44?AssignVariableOp_45?AssignVariableOp_46?AssignVariableOp_47?AssignVariableOp_48?AssignVariableOp_49?AssignVariableOp_5?AssignVariableOp_50?AssignVariableOp_51?AssignVariableOp_52?AssignVariableOp_53?AssignVariableOp_54?AssignVariableOp_6?AssignVariableOp_7?AssignVariableOp_8?AssignVariableOp_9?
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names?
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
RestoreV2/shape_and_slices?
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*?
_output_shapes?
?::::::::::::::::::::::::::::::::::::::::::::::::::::::::*F
dtypes<
:28	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0	*
_output_shapes
:2

Identity?
AssignVariableOpAssignVariableOpassignvariableop_adam_iterIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype0	2
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1?
AssignVariableOp_1AssignVariableOpassignvariableop_1_adam_beta_1Identity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2?
AssignVariableOp_2AssignVariableOpassignvariableop_2_adam_beta_2Identity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3?
AssignVariableOp_3AssignVariableOpassignvariableop_3_adam_decayIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4?
AssignVariableOp_4AssignVariableOp%assignvariableop_4_adam_learning_rateIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5?
AssignVariableOp_5AssignVariableOp9assignvariableop_5_var_auto_encoder_encoder_conv2d_kernelIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6?
AssignVariableOp_6AssignVariableOp7assignvariableop_6_var_auto_encoder_encoder_conv2d_biasIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7?
AssignVariableOp_7AssignVariableOp;assignvariableop_7_var_auto_encoder_encoder_conv2d_1_kernelIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_7k

Identity_8IdentityRestoreV2:tensors:8"/device:CPU:0*
T0*
_output_shapes
:2

Identity_8?
AssignVariableOp_8AssignVariableOp9assignvariableop_8_var_auto_encoder_encoder_conv2d_1_biasIdentity_8:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_8k

Identity_9IdentityRestoreV2:tensors:9"/device:CPU:0*
T0*
_output_shapes
:2

Identity_9?
AssignVariableOp_9AssignVariableOp8assignvariableop_9_var_auto_encoder_encoder_dense_kernelIdentity_9:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_9n
Identity_10IdentityRestoreV2:tensors:10"/device:CPU:0*
T0*
_output_shapes
:2
Identity_10?
AssignVariableOp_10AssignVariableOp7assignvariableop_10_var_auto_encoder_encoder_dense_biasIdentity_10:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_10n
Identity_11IdentityRestoreV2:tensors:11"/device:CPU:0*
T0*
_output_shapes
:2
Identity_11?
AssignVariableOp_11AssignVariableOp;assignvariableop_11_var_auto_encoder_encoder_dense_1_kernelIdentity_11:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_11n
Identity_12IdentityRestoreV2:tensors:12"/device:CPU:0*
T0*
_output_shapes
:2
Identity_12?
AssignVariableOp_12AssignVariableOp9assignvariableop_12_var_auto_encoder_encoder_dense_1_biasIdentity_12:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_12n
Identity_13IdentityRestoreV2:tensors:13"/device:CPU:0*
T0*
_output_shapes
:2
Identity_13?
AssignVariableOp_13AssignVariableOp;assignvariableop_13_var_auto_encoder_decoder_dense_2_kernelIdentity_13:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_13n
Identity_14IdentityRestoreV2:tensors:14"/device:CPU:0*
T0*
_output_shapes
:2
Identity_14?
AssignVariableOp_14AssignVariableOp9assignvariableop_14_var_auto_encoder_decoder_dense_2_biasIdentity_14:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_14n
Identity_15IdentityRestoreV2:tensors:15"/device:CPU:0*
T0*
_output_shapes
:2
Identity_15?
AssignVariableOp_15AssignVariableOpDassignvariableop_15_var_auto_encoder_decoder_conv2d_transpose_kernelIdentity_15:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_15n
Identity_16IdentityRestoreV2:tensors:16"/device:CPU:0*
T0*
_output_shapes
:2
Identity_16?
AssignVariableOp_16AssignVariableOpBassignvariableop_16_var_auto_encoder_decoder_conv2d_transpose_biasIdentity_16:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_16n
Identity_17IdentityRestoreV2:tensors:17"/device:CPU:0*
T0*
_output_shapes
:2
Identity_17?
AssignVariableOp_17AssignVariableOpFassignvariableop_17_var_auto_encoder_decoder_conv2d_transpose_1_kernelIdentity_17:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_17n
Identity_18IdentityRestoreV2:tensors:18"/device:CPU:0*
T0*
_output_shapes
:2
Identity_18?
AssignVariableOp_18AssignVariableOpDassignvariableop_18_var_auto_encoder_decoder_conv2d_transpose_1_biasIdentity_18:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_18n
Identity_19IdentityRestoreV2:tensors:19"/device:CPU:0*
T0*
_output_shapes
:2
Identity_19?
AssignVariableOp_19AssignVariableOpFassignvariableop_19_var_auto_encoder_decoder_conv2d_transpose_2_kernelIdentity_19:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_19n
Identity_20IdentityRestoreV2:tensors:20"/device:CPU:0*
T0*
_output_shapes
:2
Identity_20?
AssignVariableOp_20AssignVariableOpDassignvariableop_20_var_auto_encoder_decoder_conv2d_transpose_2_biasIdentity_20:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_20n
Identity_21IdentityRestoreV2:tensors:21"/device:CPU:0*
T0*
_output_shapes
:2
Identity_21?
AssignVariableOp_21AssignVariableOpassignvariableop_21_totalIdentity_21:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_21n
Identity_22IdentityRestoreV2:tensors:22"/device:CPU:0*
T0*
_output_shapes
:2
Identity_22?
AssignVariableOp_22AssignVariableOpassignvariableop_22_countIdentity_22:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_22n
Identity_23IdentityRestoreV2:tensors:23"/device:CPU:0*
T0*
_output_shapes
:2
Identity_23?
AssignVariableOp_23AssignVariableOpAassignvariableop_23_adam_var_auto_encoder_encoder_conv2d_kernel_mIdentity_23:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_23n
Identity_24IdentityRestoreV2:tensors:24"/device:CPU:0*
T0*
_output_shapes
:2
Identity_24?
AssignVariableOp_24AssignVariableOp?assignvariableop_24_adam_var_auto_encoder_encoder_conv2d_bias_mIdentity_24:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_24n
Identity_25IdentityRestoreV2:tensors:25"/device:CPU:0*
T0*
_output_shapes
:2
Identity_25?
AssignVariableOp_25AssignVariableOpCassignvariableop_25_adam_var_auto_encoder_encoder_conv2d_1_kernel_mIdentity_25:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_25n
Identity_26IdentityRestoreV2:tensors:26"/device:CPU:0*
T0*
_output_shapes
:2
Identity_26?
AssignVariableOp_26AssignVariableOpAassignvariableop_26_adam_var_auto_encoder_encoder_conv2d_1_bias_mIdentity_26:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_26n
Identity_27IdentityRestoreV2:tensors:27"/device:CPU:0*
T0*
_output_shapes
:2
Identity_27?
AssignVariableOp_27AssignVariableOp@assignvariableop_27_adam_var_auto_encoder_encoder_dense_kernel_mIdentity_27:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_27n
Identity_28IdentityRestoreV2:tensors:28"/device:CPU:0*
T0*
_output_shapes
:2
Identity_28?
AssignVariableOp_28AssignVariableOp>assignvariableop_28_adam_var_auto_encoder_encoder_dense_bias_mIdentity_28:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_28n
Identity_29IdentityRestoreV2:tensors:29"/device:CPU:0*
T0*
_output_shapes
:2
Identity_29?
AssignVariableOp_29AssignVariableOpBassignvariableop_29_adam_var_auto_encoder_encoder_dense_1_kernel_mIdentity_29:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_29n
Identity_30IdentityRestoreV2:tensors:30"/device:CPU:0*
T0*
_output_shapes
:2
Identity_30?
AssignVariableOp_30AssignVariableOp@assignvariableop_30_adam_var_auto_encoder_encoder_dense_1_bias_mIdentity_30:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_30n
Identity_31IdentityRestoreV2:tensors:31"/device:CPU:0*
T0*
_output_shapes
:2
Identity_31?
AssignVariableOp_31AssignVariableOpBassignvariableop_31_adam_var_auto_encoder_decoder_dense_2_kernel_mIdentity_31:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_31n
Identity_32IdentityRestoreV2:tensors:32"/device:CPU:0*
T0*
_output_shapes
:2
Identity_32?
AssignVariableOp_32AssignVariableOp@assignvariableop_32_adam_var_auto_encoder_decoder_dense_2_bias_mIdentity_32:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_32n
Identity_33IdentityRestoreV2:tensors:33"/device:CPU:0*
T0*
_output_shapes
:2
Identity_33?
AssignVariableOp_33AssignVariableOpKassignvariableop_33_adam_var_auto_encoder_decoder_conv2d_transpose_kernel_mIdentity_33:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_33n
Identity_34IdentityRestoreV2:tensors:34"/device:CPU:0*
T0*
_output_shapes
:2
Identity_34?
AssignVariableOp_34AssignVariableOpIassignvariableop_34_adam_var_auto_encoder_decoder_conv2d_transpose_bias_mIdentity_34:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_34n
Identity_35IdentityRestoreV2:tensors:35"/device:CPU:0*
T0*
_output_shapes
:2
Identity_35?
AssignVariableOp_35AssignVariableOpMassignvariableop_35_adam_var_auto_encoder_decoder_conv2d_transpose_1_kernel_mIdentity_35:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_35n
Identity_36IdentityRestoreV2:tensors:36"/device:CPU:0*
T0*
_output_shapes
:2
Identity_36?
AssignVariableOp_36AssignVariableOpKassignvariableop_36_adam_var_auto_encoder_decoder_conv2d_transpose_1_bias_mIdentity_36:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_36n
Identity_37IdentityRestoreV2:tensors:37"/device:CPU:0*
T0*
_output_shapes
:2
Identity_37?
AssignVariableOp_37AssignVariableOpMassignvariableop_37_adam_var_auto_encoder_decoder_conv2d_transpose_2_kernel_mIdentity_37:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_37n
Identity_38IdentityRestoreV2:tensors:38"/device:CPU:0*
T0*
_output_shapes
:2
Identity_38?
AssignVariableOp_38AssignVariableOpKassignvariableop_38_adam_var_auto_encoder_decoder_conv2d_transpose_2_bias_mIdentity_38:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_38n
Identity_39IdentityRestoreV2:tensors:39"/device:CPU:0*
T0*
_output_shapes
:2
Identity_39?
AssignVariableOp_39AssignVariableOpAassignvariableop_39_adam_var_auto_encoder_encoder_conv2d_kernel_vIdentity_39:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_39n
Identity_40IdentityRestoreV2:tensors:40"/device:CPU:0*
T0*
_output_shapes
:2
Identity_40?
AssignVariableOp_40AssignVariableOp?assignvariableop_40_adam_var_auto_encoder_encoder_conv2d_bias_vIdentity_40:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_40n
Identity_41IdentityRestoreV2:tensors:41"/device:CPU:0*
T0*
_output_shapes
:2
Identity_41?
AssignVariableOp_41AssignVariableOpCassignvariableop_41_adam_var_auto_encoder_encoder_conv2d_1_kernel_vIdentity_41:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_41n
Identity_42IdentityRestoreV2:tensors:42"/device:CPU:0*
T0*
_output_shapes
:2
Identity_42?
AssignVariableOp_42AssignVariableOpAassignvariableop_42_adam_var_auto_encoder_encoder_conv2d_1_bias_vIdentity_42:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_42n
Identity_43IdentityRestoreV2:tensors:43"/device:CPU:0*
T0*
_output_shapes
:2
Identity_43?
AssignVariableOp_43AssignVariableOp@assignvariableop_43_adam_var_auto_encoder_encoder_dense_kernel_vIdentity_43:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_43n
Identity_44IdentityRestoreV2:tensors:44"/device:CPU:0*
T0*
_output_shapes
:2
Identity_44?
AssignVariableOp_44AssignVariableOp>assignvariableop_44_adam_var_auto_encoder_encoder_dense_bias_vIdentity_44:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_44n
Identity_45IdentityRestoreV2:tensors:45"/device:CPU:0*
T0*
_output_shapes
:2
Identity_45?
AssignVariableOp_45AssignVariableOpBassignvariableop_45_adam_var_auto_encoder_encoder_dense_1_kernel_vIdentity_45:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_45n
Identity_46IdentityRestoreV2:tensors:46"/device:CPU:0*
T0*
_output_shapes
:2
Identity_46?
AssignVariableOp_46AssignVariableOp@assignvariableop_46_adam_var_auto_encoder_encoder_dense_1_bias_vIdentity_46:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_46n
Identity_47IdentityRestoreV2:tensors:47"/device:CPU:0*
T0*
_output_shapes
:2
Identity_47?
AssignVariableOp_47AssignVariableOpBassignvariableop_47_adam_var_auto_encoder_decoder_dense_2_kernel_vIdentity_47:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_47n
Identity_48IdentityRestoreV2:tensors:48"/device:CPU:0*
T0*
_output_shapes
:2
Identity_48?
AssignVariableOp_48AssignVariableOp@assignvariableop_48_adam_var_auto_encoder_decoder_dense_2_bias_vIdentity_48:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_48n
Identity_49IdentityRestoreV2:tensors:49"/device:CPU:0*
T0*
_output_shapes
:2
Identity_49?
AssignVariableOp_49AssignVariableOpKassignvariableop_49_adam_var_auto_encoder_decoder_conv2d_transpose_kernel_vIdentity_49:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_49n
Identity_50IdentityRestoreV2:tensors:50"/device:CPU:0*
T0*
_output_shapes
:2
Identity_50?
AssignVariableOp_50AssignVariableOpIassignvariableop_50_adam_var_auto_encoder_decoder_conv2d_transpose_bias_vIdentity_50:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_50n
Identity_51IdentityRestoreV2:tensors:51"/device:CPU:0*
T0*
_output_shapes
:2
Identity_51?
AssignVariableOp_51AssignVariableOpMassignvariableop_51_adam_var_auto_encoder_decoder_conv2d_transpose_1_kernel_vIdentity_51:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_51n
Identity_52IdentityRestoreV2:tensors:52"/device:CPU:0*
T0*
_output_shapes
:2
Identity_52?
AssignVariableOp_52AssignVariableOpKassignvariableop_52_adam_var_auto_encoder_decoder_conv2d_transpose_1_bias_vIdentity_52:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_52n
Identity_53IdentityRestoreV2:tensors:53"/device:CPU:0*
T0*
_output_shapes
:2
Identity_53?
AssignVariableOp_53AssignVariableOpMassignvariableop_53_adam_var_auto_encoder_decoder_conv2d_transpose_2_kernel_vIdentity_53:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_53n
Identity_54IdentityRestoreV2:tensors:54"/device:CPU:0*
T0*
_output_shapes
:2
Identity_54?
AssignVariableOp_54AssignVariableOpKassignvariableop_54_adam_var_auto_encoder_decoder_conv2d_transpose_2_bias_vIdentity_54:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_549
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp?

Identity_55Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2
Identity_55?

Identity_56IdentityIdentity_55:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_10^AssignVariableOp_11^AssignVariableOp_12^AssignVariableOp_13^AssignVariableOp_14^AssignVariableOp_15^AssignVariableOp_16^AssignVariableOp_17^AssignVariableOp_18^AssignVariableOp_19^AssignVariableOp_2^AssignVariableOp_20^AssignVariableOp_21^AssignVariableOp_22^AssignVariableOp_23^AssignVariableOp_24^AssignVariableOp_25^AssignVariableOp_26^AssignVariableOp_27^AssignVariableOp_28^AssignVariableOp_29^AssignVariableOp_3^AssignVariableOp_30^AssignVariableOp_31^AssignVariableOp_32^AssignVariableOp_33^AssignVariableOp_34^AssignVariableOp_35^AssignVariableOp_36^AssignVariableOp_37^AssignVariableOp_38^AssignVariableOp_39^AssignVariableOp_4^AssignVariableOp_40^AssignVariableOp_41^AssignVariableOp_42^AssignVariableOp_43^AssignVariableOp_44^AssignVariableOp_45^AssignVariableOp_46^AssignVariableOp_47^AssignVariableOp_48^AssignVariableOp_49^AssignVariableOp_5^AssignVariableOp_50^AssignVariableOp_51^AssignVariableOp_52^AssignVariableOp_53^AssignVariableOp_54^AssignVariableOp_6^AssignVariableOp_7^AssignVariableOp_8^AssignVariableOp_9*
T0*
_output_shapes
: 2
Identity_56"#
identity_56Identity_56:output:0*?
_input_shapes?
?: :::::::::::::::::::::::::::::::::::::::::::::::::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12*
AssignVariableOp_10AssignVariableOp_102*
AssignVariableOp_11AssignVariableOp_112*
AssignVariableOp_12AssignVariableOp_122*
AssignVariableOp_13AssignVariableOp_132*
AssignVariableOp_14AssignVariableOp_142*
AssignVariableOp_15AssignVariableOp_152*
AssignVariableOp_16AssignVariableOp_162*
AssignVariableOp_17AssignVariableOp_172*
AssignVariableOp_18AssignVariableOp_182*
AssignVariableOp_19AssignVariableOp_192(
AssignVariableOp_2AssignVariableOp_22*
AssignVariableOp_20AssignVariableOp_202*
AssignVariableOp_21AssignVariableOp_212*
AssignVariableOp_22AssignVariableOp_222*
AssignVariableOp_23AssignVariableOp_232*
AssignVariableOp_24AssignVariableOp_242*
AssignVariableOp_25AssignVariableOp_252*
AssignVariableOp_26AssignVariableOp_262*
AssignVariableOp_27AssignVariableOp_272*
AssignVariableOp_28AssignVariableOp_282*
AssignVariableOp_29AssignVariableOp_292(
AssignVariableOp_3AssignVariableOp_32*
AssignVariableOp_30AssignVariableOp_302*
AssignVariableOp_31AssignVariableOp_312*
AssignVariableOp_32AssignVariableOp_322*
AssignVariableOp_33AssignVariableOp_332*
AssignVariableOp_34AssignVariableOp_342*
AssignVariableOp_35AssignVariableOp_352*
AssignVariableOp_36AssignVariableOp_362*
AssignVariableOp_37AssignVariableOp_372*
AssignVariableOp_38AssignVariableOp_382*
AssignVariableOp_39AssignVariableOp_392(
AssignVariableOp_4AssignVariableOp_42*
AssignVariableOp_40AssignVariableOp_402*
AssignVariableOp_41AssignVariableOp_412*
AssignVariableOp_42AssignVariableOp_422*
AssignVariableOp_43AssignVariableOp_432*
AssignVariableOp_44AssignVariableOp_442*
AssignVariableOp_45AssignVariableOp_452*
AssignVariableOp_46AssignVariableOp_462*
AssignVariableOp_47AssignVariableOp_472*
AssignVariableOp_48AssignVariableOp_482*
AssignVariableOp_49AssignVariableOp_492(
AssignVariableOp_5AssignVariableOp_52*
AssignVariableOp_50AssignVariableOp_502*
AssignVariableOp_51AssignVariableOp_512*
AssignVariableOp_52AssignVariableOp_522*
AssignVariableOp_53AssignVariableOp_532*
AssignVariableOp_54AssignVariableOp_542(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_72(
AssignVariableOp_8AssignVariableOp_82(
AssignVariableOp_9AssignVariableOp_9:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
?:
?
B__inference_encoder_layer_call_and_return_conditional_losses_60542

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity

identity_1

identity_2??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten/Const?
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddf
sampling/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean?
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sampling/random_normal/stddev?
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02-
+sampling/random_normal/RandomStandardNormal?
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
sampling/random_normal/mul?
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x?
sampling/mulMulsampling/mul/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling/Exp?
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling/mul_1?
sampling/addAddV2dense/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling/add?
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitysampling/add:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*N
_input_shapes=
;:?????????::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_decoder_layer_call_fn_60773

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_595862
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
7
 __inference_log_normal_pdf_59178
z
identityS
Log/xConst*
_output_shapes
: *
dtype0*
valueB
 *??@2
Log/xB
LogLogLog/x:output:0*
T0*
_output_shapes
: 2
LogS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sub/yV
subSubzsub/y:output:0*
T0*'
_output_shapes
:?????????2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowsub:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powS
Exp/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Exp/xB
ExpExpExp/x:output:0*
T0*
_output_shapes
: 2
ExpU
mulMulpow:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/y^
addAddV2mul:z:0add/y:output:0*
T0*'
_output_shapes
:?????????2
add[
add_1AddV2add:z:0Log:y:0*
T0*'
_output_shapes
:?????????2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xd
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
mul_1p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_1:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum\
IdentityIdentitySum:output:0*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:J F
'
_output_shapes
:?????????

_user_specified_namez
?
7
 __inference_log_normal_pdf_60432
z
identityS
Log/xConst*
_output_shapes
: *
dtype0*
valueB
 *??@2
Log/xB
LogLogLog/x:output:0*
T0*
_output_shapes
: 2
LogS
sub/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sub/yV
subSubzsub/y:output:0*
T0*'
_output_shapes
:?????????2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowsub:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powS
Exp/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
Exp/xB
ExpExpExp/x:output:0*
T0*
_output_shapes
: 2
ExpU
mulMulpow:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulS
add/yConst*
_output_shapes
: *
dtype0*
valueB
 *    2
add/y^
addAddV2mul:z:0add/y:output:0*
T0*'
_output_shapes
:?????????2
add[
add_1AddV2add:z:0Log:y:0*
T0*'
_output_shapes
:?????????2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xd
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
mul_1p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_1:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum\
IdentityIdentitySum:output:0*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*&
_input_shapes
:?????????:J F
'
_output_shapes
:?????????

_user_specified_namez
?
?
2__inference_conv2d_transpose_2_layer_call_fn_59341

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+???????????????????????????*$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_593312
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?+
?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_59810
input_1
encoder_59752
encoder_59754
encoder_59756
encoder_59758
encoder_59760
encoder_59762
encoder_59764
encoder_59766
decoder_59771
decoder_59773
decoder_59775
decoder_59777
decoder_59779
decoder_59781
decoder_59783
decoder_59785
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_59752encoder_59754encoder_59756encoder_59758encoder_59760encoder_59762encoder_59764encoder_59766*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_594352!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:2decoder_59771decoder_59773decoder_59775decoder_59777decoder_59779decoder_59781decoder_59783decoder_59785*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_596662!
decoder/StatefulPartitionedCall?
logistic_loss/zeros_like	ZerosLike(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/zeros_like?
logistic_loss/GreaterEqualGreaterEqual(decoder/StatefulPartitionedCall:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/GreaterEqual?
logistic_loss/SelectSelectlogistic_loss/GreaterEqual:z:0(decoder/StatefulPartitionedCall:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select?
logistic_loss/NegNeg(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Neg?
logistic_loss/Select_1Selectlogistic_loss/GreaterEqual:z:0logistic_loss/Neg:y:0(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select_1?
logistic_loss/mulMul(decoder/StatefulPartitionedCall:output:0input_1*
T0*/
_output_shapes
:?????????2
logistic_loss/mul?
logistic_loss/subSublogistic_loss/Select:output:0logistic_loss/mul:z:0*
T0*/
_output_shapes
:?????????2
logistic_loss/sub?
logistic_loss/ExpExplogistic_loss/Select_1:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Exp?
logistic_loss/Log1pLog1plogistic_loss/Exp:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Log1p?
logistic_lossAddlogistic_loss/sub:z:0logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss?
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Sum/reduction_indicesr
SumSumlogistic_loss:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumM
NegNegSum:output:0*
T0*#
_output_shapes
:?????????2
Neg?
PartitionedCallPartitionedCall(encoder/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591782
PartitionedCall?
PartitionedCall_1PartitionedCall(encoder/StatefulPartitionedCall:output:2(encoder/StatefulPartitionedCall:output:0(encoder/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591992
PartitionedCall_1u
subSubPartitionedCall:output:0PartitionedCall_1:output:0*
T0*#
_output_shapes
:?????????2
subS
addAddV2Neg:y:0sub:z:0*
T0*#
_output_shapes
:?????????2
addX
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstN
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: 2
MeanE
Neg_1NegMean:output:0*
T0*
_output_shapes
: 2
Neg_1?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity	Neg_1:y:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?
M
 __inference_log_normal_pdf_59199
z
mean

logvar
identityS
Log/xConst*
_output_shapes
: *
dtype0*
valueB
 *??@2
Log/xB
LogLogLog/x:output:0*
T0*
_output_shapes
: 2
LogL
subSubzmean*
T0*'
_output_shapes
:?????????2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowsub:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powK
NegNeglogvar*
T0*'
_output_shapes
:?????????2
NegL
ExpExpNeg:y:0*
T0*'
_output_shapes
:?????????2
ExpU
mulMulpow:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulV
addAddV2mul:z:0logvar*
T0*'
_output_shapes
:?????????2
add[
add_1AddV2add:z:0Log:y:0*
T0*'
_output_shapes
:?????????2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xd
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
mul_1p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_1:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum\
IdentityIdentitySum:output:0*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:?????????:?????????:?????????:J F
'
_output_shapes
:?????????

_user_specified_namez:MI
'
_output_shapes
:?????????

_user_specified_namemean:OK
'
_output_shapes
:?????????
 
_user_specified_namelogvar
?$
?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_59242

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????@*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:@*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????@2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+???????????????????????????@2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????@2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?
?
2__inference_conv2d_transpose_1_layer_call_fn_59297

inputs
unknown
	unknown_0
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *A
_output_shapes/
-:+??????????????????????????? *$
_read_only_resource_inputs
*0
config_proto 

CPU

GPU2*0J 8? *V
fQRO
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_592872
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::22
StatefulPartitionedCallStatefulPartitionedCall:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
?+
?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_59749
input_1
encoder_59487
encoder_59489
encoder_59491
encoder_59493
encoder_59495
encoder_59497
encoder_59499
encoder_59501
decoder_59710
decoder_59712
decoder_59714
decoder_59716
decoder_59718
decoder_59720
decoder_59722
decoder_59724
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinput_1encoder_59487encoder_59489encoder_59491encoder_59493encoder_59495encoder_59497encoder_59499encoder_59501*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_593902!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:2decoder_59710decoder_59712decoder_59714decoder_59716decoder_59718decoder_59720decoder_59722decoder_59724*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_595862!
decoder/StatefulPartitionedCall?
logistic_loss/zeros_like	ZerosLike(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/zeros_like?
logistic_loss/GreaterEqualGreaterEqual(decoder/StatefulPartitionedCall:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/GreaterEqual?
logistic_loss/SelectSelectlogistic_loss/GreaterEqual:z:0(decoder/StatefulPartitionedCall:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select?
logistic_loss/NegNeg(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Neg?
logistic_loss/Select_1Selectlogistic_loss/GreaterEqual:z:0logistic_loss/Neg:y:0(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select_1?
logistic_loss/mulMul(decoder/StatefulPartitionedCall:output:0input_1*
T0*/
_output_shapes
:?????????2
logistic_loss/mul?
logistic_loss/subSublogistic_loss/Select:output:0logistic_loss/mul:z:0*
T0*/
_output_shapes
:?????????2
logistic_loss/sub?
logistic_loss/ExpExplogistic_loss/Select_1:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Exp?
logistic_loss/Log1pLog1plogistic_loss/Exp:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Log1p?
logistic_lossAddlogistic_loss/sub:z:0logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss?
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Sum/reduction_indicesr
SumSumlogistic_loss:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumM
NegNegSum:output:0*
T0*#
_output_shapes
:?????????2
Neg?
PartitionedCallPartitionedCall(encoder/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591782
PartitionedCall?
PartitionedCall_1PartitionedCall(encoder/StatefulPartitionedCall:output:2(encoder/StatefulPartitionedCall:output:0(encoder/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591992
PartitionedCall_1u
subSubPartitionedCall:output:0PartitionedCall_1:output:0*
T0*#
_output_shapes
:?????????2
subS
addAddV2Neg:y:0sub:z:0*
T0*#
_output_shapes
:?????????2
addX
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstN
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: 2
MeanE
Neg_1NegMean:output:0*
T0*
_output_shapes
: 2
Neg_1?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity	Neg_1:y:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?u
?
B__inference_decoder_layer_call_and_return_conditional_losses_60752

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Reluh
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose/BiasAdd?
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose/Relu?
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/Relu?
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_2/BiasAdd?
IdentityIdentity#conv2d_transpose_2/BiasAdd:output:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
0__inference_var_auto_encoder_layer_call_fn_60374

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_598742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_decoder_layer_call_fn_60794

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_596662
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?u
?
B__inference_decoder_layer_call_and_return_conditional_losses_59586

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Reluh
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose/BiasAdd?
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose/Relu?
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/Relu?
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_2/BiasAdd?
IdentityIdentity#conv2d_transpose_2/BiasAdd:output:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_encoder_layer_call_fn_60567

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_593902
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?
M
 __inference_log_normal_pdf_60452
z
mean

logvar
identityS
Log/xConst*
_output_shapes
: *
dtype0*
valueB
 *??@2
Log/xB
LogLogLog/x:output:0*
T0*
_output_shapes
: 2
LogL
subSubzmean*
T0*'
_output_shapes
:?????????2
subS
pow/yConst*
_output_shapes
: *
dtype0*
valueB
 *   @2
pow/y\
powPowsub:z:0pow/y:output:0*
T0*'
_output_shapes
:?????????2
powK
NegNeglogvar*
T0*'
_output_shapes
:?????????2
NegL
ExpExpNeg:y:0*
T0*'
_output_shapes
:?????????2
ExpU
mulMulpow:z:0Exp:y:0*
T0*'
_output_shapes
:?????????2
mulV
addAddV2mul:z:0logvar*
T0*'
_output_shapes
:?????????2
add[
add_1AddV2add:z:0Log:y:0*
T0*'
_output_shapes
:?????????2
add_1W
mul_1/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2	
mul_1/xd
mul_1Mulmul_1/x:output:0	add_1:z:0*
T0*'
_output_shapes
:?????????2
mul_1p
Sum/reduction_indicesConst*
_output_shapes
: *
dtype0*
value	B :2
Sum/reduction_indicesj
SumSum	mul_1:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
Sum\
IdentityIdentitySum:output:0*
T0*#
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*L
_input_shapes;
9:?????????:?????????:?????????:J F
'
_output_shapes
:?????????

_user_specified_namez:MI
'
_output_shapes
:?????????

_user_specified_namemean:OK
'
_output_shapes
:?????????
 
_user_specified_namelogvar
?#
?
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_59331

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B :2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+???????????????????????????*
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+???????????????????????????2	
BiasAdd?
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+???????????????????????????2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+??????????????????????????? ::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+??????????????????????????? 
 
_user_specified_nameinputs
?+
?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_59874

inputs
encoder_59816
encoder_59818
encoder_59820
encoder_59822
encoder_59824
encoder_59826
encoder_59828
encoder_59830
decoder_59835
decoder_59837
decoder_59839
decoder_59841
decoder_59843
decoder_59845
decoder_59847
decoder_59849
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_59816encoder_59818encoder_59820encoder_59822encoder_59824encoder_59826encoder_59828encoder_59830*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_593902!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:2decoder_59835decoder_59837decoder_59839decoder_59841decoder_59843decoder_59845decoder_59847decoder_59849*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_595862!
decoder/StatefulPartitionedCall?
logistic_loss/zeros_like	ZerosLike(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/zeros_like?
logistic_loss/GreaterEqualGreaterEqual(decoder/StatefulPartitionedCall:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/GreaterEqual?
logistic_loss/SelectSelectlogistic_loss/GreaterEqual:z:0(decoder/StatefulPartitionedCall:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select?
logistic_loss/NegNeg(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Neg?
logistic_loss/Select_1Selectlogistic_loss/GreaterEqual:z:0logistic_loss/Neg:y:0(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select_1?
logistic_loss/mulMul(decoder/StatefulPartitionedCall:output:0inputs*
T0*/
_output_shapes
:?????????2
logistic_loss/mul?
logistic_loss/subSublogistic_loss/Select:output:0logistic_loss/mul:z:0*
T0*/
_output_shapes
:?????????2
logistic_loss/sub?
logistic_loss/ExpExplogistic_loss/Select_1:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Exp?
logistic_loss/Log1pLog1plogistic_loss/Exp:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Log1p?
logistic_lossAddlogistic_loss/sub:z:0logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss?
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Sum/reduction_indicesr
SumSumlogistic_loss:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumM
NegNegSum:output:0*
T0*#
_output_shapes
:?????????2
Neg?
PartitionedCallPartitionedCall(encoder/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591782
PartitionedCall?
PartitionedCall_1PartitionedCall(encoder/StatefulPartitionedCall:output:2(encoder/StatefulPartitionedCall:output:0(encoder/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591992
PartitionedCall_1u
subSubPartitionedCall:output:0PartitionedCall_1:output:0*
T0*#
_output_shapes
:?????????2
subS
addAddV2Neg:y:0sub:z:0*
T0*#
_output_shapes
:?????????2
addX
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstN
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: 2
MeanE
Neg_1NegMean:output:0*
T0*
_output_shapes
: 2
Neg_1?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity	Neg_1:y:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
#__inference_signature_wrapper_60056
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????*2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference__wrapped_model_592072
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?u
?
B__inference_decoder_layer_call_and_return_conditional_losses_60672

inputs*
&dense_2_matmul_readvariableop_resource+
'dense_2_biasadd_readvariableop_resource=
9conv2d_transpose_conv2d_transpose_readvariableop_resource4
0conv2d_transpose_biasadd_readvariableop_resource?
;conv2d_transpose_1_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_1_biasadd_readvariableop_resource?
;conv2d_transpose_2_conv2d_transpose_readvariableop_resource6
2conv2d_transpose_2_biasadd_readvariableop_resource
identity??'conv2d_transpose/BiasAdd/ReadVariableOp?0conv2d_transpose/conv2d_transpose/ReadVariableOp?)conv2d_transpose_1/BiasAdd/ReadVariableOp?2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?)conv2d_transpose_2/BiasAdd/ReadVariableOp?2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?dense_2/BiasAdd/ReadVariableOp?dense_2/MatMul/ReadVariableOp?
dense_2/MatMul/ReadVariableOpReadVariableOp&dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_2/MatMul/ReadVariableOp?
dense_2/MatMulMatMulinputs%dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/MatMul?
dense_2/BiasAdd/ReadVariableOpReadVariableOp'dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02 
dense_2/BiasAdd/ReadVariableOp?
dense_2/BiasAddBiasAdddense_2/MatMul:product:0&dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
dense_2/BiasAddq
dense_2/ReluReludense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
dense_2/Reluh
reshape/ShapeShapedense_2/Relu:activations:0*
T0*
_output_shapes
:2
reshape/Shape?
reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
reshape/strided_slice/stack?
reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_1?
reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
reshape/strided_slice/stack_2?
reshape/strided_sliceStridedSlicereshape/Shape:output:0$reshape/strided_slice/stack:output:0&reshape/strided_slice/stack_1:output:0&reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
reshape/strided_slicet
reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/1t
reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2
reshape/Reshape/shape/2t
reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2
reshape/Reshape/shape/3?
reshape/Reshape/shapePackreshape/strided_slice:output:0 reshape/Reshape/shape/1:output:0 reshape/Reshape/shape/2:output:0 reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
reshape/Reshape/shape?
reshape/ReshapeReshapedense_2/Relu:activations:0reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
reshape/Reshapex
conv2d_transpose/ShapeShapereshape/Reshape:output:0*
T0*
_output_shapes
:2
conv2d_transpose/Shape?
$conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2&
$conv2d_transpose/strided_slice/stack?
&conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_1?
&conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2(
&conv2d_transpose/strided_slice/stack_2?
conv2d_transpose/strided_sliceStridedSliceconv2d_transpose/Shape:output:0-conv2d_transpose/strided_slice/stack:output:0/conv2d_transpose/strided_slice/stack_1:output:0/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2 
conv2d_transpose/strided_slicev
conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/1v
conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose/stack/2v
conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2
conv2d_transpose/stack/3?
conv2d_transpose/stackPack'conv2d_transpose/strided_slice:output:0!conv2d_transpose/stack/1:output:0!conv2d_transpose/stack/2:output:0!conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose/stack?
&conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose/strided_slice_1/stack?
(conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_1?
(conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose/strided_slice_1/stack_2?
 conv2d_transpose/strided_slice_1StridedSliceconv2d_transpose/stack:output:0/conv2d_transpose/strided_slice_1/stack:output:01conv2d_transpose/strided_slice_1/stack_1:output:01conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose/strided_slice_1?
0conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOp9conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype022
0conv2d_transpose/conv2d_transpose/ReadVariableOp?
!conv2d_transpose/conv2d_transposeConv2DBackpropInputconv2d_transpose/stack:output:08conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2#
!conv2d_transpose/conv2d_transpose?
'conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp0conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'conv2d_transpose/BiasAdd/ReadVariableOp?
conv2d_transpose/BiasAddBiasAdd*conv2d_transpose/conv2d_transpose:output:0/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose/BiasAdd?
conv2d_transpose/ReluRelu!conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_transpose/Relu?
conv2d_transpose_1/ShapeShape#conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_1/Shape?
&conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_1/strided_slice/stack?
(conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_1?
(conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_1/strided_slice/stack_2?
 conv2d_transpose_1/strided_sliceStridedSlice!conv2d_transpose_1/Shape:output:0/conv2d_transpose_1/strided_slice/stack:output:01conv2d_transpose_1/strided_slice/stack_1:output:01conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_1/strided_slicez
conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/1z
conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_1/stack/2z
conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2
conv2d_transpose_1/stack/3?
conv2d_transpose_1/stackPack)conv2d_transpose_1/strided_slice:output:0#conv2d_transpose_1/stack/1:output:0#conv2d_transpose_1/stack/2:output:0#conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_1/stack?
(conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_1/strided_slice_1/stack?
*conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_1?
*conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_1/strided_slice_1/stack_2?
"conv2d_transpose_1/strided_slice_1StridedSlice!conv2d_transpose_1/stack:output:01conv2d_transpose_1/strided_slice_1/stack:output:03conv2d_transpose_1/strided_slice_1/stack_1:output:03conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_1/strided_slice_1?
2conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype024
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_1/conv2d_transposeConv2DBackpropInput!conv2d_transpose_1/stack:output:0:conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0#conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2%
#conv2d_transpose_1/conv2d_transpose?
)conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02+
)conv2d_transpose_1/BiasAdd/ReadVariableOp?
conv2d_transpose_1/BiasAddBiasAdd,conv2d_transpose_1/conv2d_transpose:output:01conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/BiasAdd?
conv2d_transpose_1/ReluRelu#conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d_transpose_1/Relu?
conv2d_transpose_2/ShapeShape%conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2
conv2d_transpose_2/Shape?
&conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2(
&conv2d_transpose_2/strided_slice/stack?
(conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_1?
(conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2*
(conv2d_transpose_2/strided_slice/stack_2?
 conv2d_transpose_2/strided_sliceStridedSlice!conv2d_transpose_2/Shape:output:0/conv2d_transpose_2/strided_slice/stack:output:01conv2d_transpose_2/strided_slice/stack_1:output:01conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2"
 conv2d_transpose_2/strided_slicez
conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/1z
conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/2z
conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2
conv2d_transpose_2/stack/3?
conv2d_transpose_2/stackPack)conv2d_transpose_2/strided_slice:output:0#conv2d_transpose_2/stack/1:output:0#conv2d_transpose_2/stack/2:output:0#conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2
conv2d_transpose_2/stack?
(conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2*
(conv2d_transpose_2/strided_slice_1/stack?
*conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_1?
*conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2,
*conv2d_transpose_2/strided_slice_1/stack_2?
"conv2d_transpose_2/strided_slice_1StridedSlice!conv2d_transpose_2/stack:output:01conv2d_transpose_2/strided_slice_1/stack:output:03conv2d_transpose_2/strided_slice_1/stack_1:output:03conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2$
"conv2d_transpose_2/strided_slice_1?
2conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOp;conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype024
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
#conv2d_transpose_2/conv2d_transposeConv2DBackpropInput!conv2d_transpose_2/stack:output:0:conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0%conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2%
#conv2d_transpose_2/conv2d_transpose?
)conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp2conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02+
)conv2d_transpose_2/BiasAdd/ReadVariableOp?
conv2d_transpose_2/BiasAddBiasAdd,conv2d_transpose_2/conv2d_transpose:output:01conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2
conv2d_transpose_2/BiasAdd?
IdentityIdentity#conv2d_transpose_2/BiasAdd:output:0(^conv2d_transpose/BiasAdd/ReadVariableOp1^conv2d_transpose/conv2d_transpose/ReadVariableOp*^conv2d_transpose_1/BiasAdd/ReadVariableOp3^conv2d_transpose_1/conv2d_transpose/ReadVariableOp*^conv2d_transpose_2/BiasAdd/ReadVariableOp3^conv2d_transpose_2/conv2d_transpose/ReadVariableOp^dense_2/BiasAdd/ReadVariableOp^dense_2/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:?????????::::::::2R
'conv2d_transpose/BiasAdd/ReadVariableOp'conv2d_transpose/BiasAdd/ReadVariableOp2d
0conv2d_transpose/conv2d_transpose/ReadVariableOp0conv2d_transpose/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_1/BiasAdd/ReadVariableOp)conv2d_transpose_1/BiasAdd/ReadVariableOp2h
2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2conv2d_transpose_1/conv2d_transpose/ReadVariableOp2V
)conv2d_transpose_2/BiasAdd/ReadVariableOp)conv2d_transpose_2/BiasAdd/ReadVariableOp2h
2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2conv2d_transpose_2/conv2d_transpose/ReadVariableOp2@
dense_2/BiasAdd/ReadVariableOpdense_2/BiasAdd/ReadVariableOp2>
dense_2/MatMul/ReadVariableOpdense_2/MatMul/ReadVariableOp:O K
'
_output_shapes
:?????????
 
_user_specified_nameinputs
?
?
'__inference_encoder_layer_call_fn_60592

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity

identity_1

identity_2??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_594352
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identity StatefulPartitionedCall:output:1^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identity StatefulPartitionedCall:output:2^StatefulPartitionedCall*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*N
_input_shapes=
;:?????????::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?:
?
B__inference_encoder_layer_call_and_return_conditional_losses_59435

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity

identity_1

identity_2??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten/Const?
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddf
sampling/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean?
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sampling/random_normal/stddev?
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02-
+sampling/random_normal/RandomStandardNormal?
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
sampling/random_normal/mul?
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x?
sampling/mulMulsampling/mul/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling/Exp?
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling/mul_1?
sampling/addAddV2dense/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling/add?
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitysampling/add:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*N
_input_shapes=
;:?????????::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?:
?
B__inference_encoder_layer_call_and_return_conditional_losses_59390

inputs)
%conv2d_conv2d_readvariableop_resource*
&conv2d_biasadd_readvariableop_resource+
'conv2d_1_conv2d_readvariableop_resource,
(conv2d_1_biasadd_readvariableop_resource(
$dense_matmul_readvariableop_resource)
%dense_biasadd_readvariableop_resource*
&dense_1_matmul_readvariableop_resource+
'dense_1_biasadd_readvariableop_resource
identity

identity_1

identity_2??conv2d/BiasAdd/ReadVariableOp?conv2d/Conv2D/ReadVariableOp?conv2d_1/BiasAdd/ReadVariableOp?conv2d_1/Conv2D/ReadVariableOp?dense/BiasAdd/ReadVariableOp?dense/MatMul/ReadVariableOp?dense_1/BiasAdd/ReadVariableOp?dense_1/MatMul/ReadVariableOp?
conv2d/Conv2D/ReadVariableOpReadVariableOp%conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02
conv2d/Conv2D/ReadVariableOp?
conv2d/Conv2DConv2Dinputs$conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
conv2d/Conv2D?
conv2d/BiasAdd/ReadVariableOpReadVariableOp&conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02
conv2d/BiasAdd/ReadVariableOp?
conv2d/BiasAddBiasAddconv2d/Conv2D:output:0%conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
conv2d/BiasAddu
conv2d/ReluReluconv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
conv2d/Relu?
conv2d_1/Conv2D/ReadVariableOpReadVariableOp'conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02 
conv2d_1/Conv2D/ReadVariableOp?
conv2d_1/Conv2DConv2Dconv2d/Relu:activations:0&conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
conv2d_1/Conv2D?
conv2d_1/BiasAdd/ReadVariableOpReadVariableOp(conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02!
conv2d_1/BiasAdd/ReadVariableOp?
conv2d_1/BiasAddBiasAddconv2d_1/Conv2D:output:0'conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/BiasAdd{
conv2d_1/ReluReluconv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
conv2d_1/Reluo
flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
flatten/Const?
flatten/ReshapeReshapeconv2d_1/Relu:activations:0flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
flatten/Reshape?
dense/MatMul/ReadVariableOpReadVariableOp$dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense/MatMul/ReadVariableOp?
dense/MatMulMatMulflatten/Reshape:output:0#dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/MatMul?
dense/BiasAdd/ReadVariableOpReadVariableOp%dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02
dense/BiasAdd/ReadVariableOp?
dense/BiasAddBiasAdddense/MatMul:product:0$dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense/BiasAdd?
dense_1/MatMul/ReadVariableOpReadVariableOp&dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02
dense_1/MatMul/ReadVariableOp?
dense_1/MatMulMatMulflatten/Reshape:output:0%dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/MatMul?
dense_1/BiasAdd/ReadVariableOpReadVariableOp'dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
dense_1/BiasAdd/ReadVariableOp?
dense_1/BiasAddBiasAdddense_1/MatMul:product:0&dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
dense_1/BiasAddf
sampling/ShapeShapedense/BiasAdd:output:0*
T0*
_output_shapes
:2
sampling/Shape
sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2
sampling/random_normal/mean?
sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2
sampling/random_normal/stddev?
+sampling/random_normal/RandomStandardNormalRandomStandardNormalsampling/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02-
+sampling/random_normal/RandomStandardNormal?
sampling/random_normal/mulMul4sampling/random_normal/RandomStandardNormal:output:0&sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2
sampling/random_normal/mul?
sampling/random_normalAddsampling/random_normal/mul:z:0$sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2
sampling/random_normale
sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
sampling/mul/x?
sampling/mulMulsampling/mul/x:output:0dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
sampling/mulg
sampling/ExpExpsampling/mul:z:0*
T0*'
_output_shapes
:?????????2
sampling/Exp?
sampling/mul_1Mulsampling/Exp:y:0sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
sampling/mul_1?
sampling/addAddV2dense/BiasAdd:output:0sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
sampling/add?
IdentityIdentitydense/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity?

Identity_1Identitydense_1/BiasAdd:output:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_1?

Identity_2Identitysampling/add:z:0^conv2d/BiasAdd/ReadVariableOp^conv2d/Conv2D/ReadVariableOp ^conv2d_1/BiasAdd/ReadVariableOp^conv2d_1/Conv2D/ReadVariableOp^dense/BiasAdd/ReadVariableOp^dense/MatMul/ReadVariableOp^dense_1/BiasAdd/ReadVariableOp^dense_1/MatMul/ReadVariableOp*
T0*'
_output_shapes
:?????????2

Identity_2"
identityIdentity:output:0"!

identity_1Identity_1:output:0"!

identity_2Identity_2:output:0*N
_input_shapes=
;:?????????::::::::2>
conv2d/BiasAdd/ReadVariableOpconv2d/BiasAdd/ReadVariableOp2<
conv2d/Conv2D/ReadVariableOpconv2d/Conv2D/ReadVariableOp2B
conv2d_1/BiasAdd/ReadVariableOpconv2d_1/BiasAdd/ReadVariableOp2@
conv2d_1/Conv2D/ReadVariableOpconv2d_1/Conv2D/ReadVariableOp2<
dense/BiasAdd/ReadVariableOpdense/BiasAdd/ReadVariableOp2:
dense/MatMul/ReadVariableOpdense/MatMul/ReadVariableOp2@
dense_1/BiasAdd/ReadVariableOpdense_1/BiasAdd/ReadVariableOp2>
dense_1/MatMul/ReadVariableOpdense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?

?
0__inference_var_auto_encoder_layer_call_fn_59910
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_598742
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
0__inference_var_auto_encoder_layer_call_fn_60009
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_599732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
??
? 
__inference__traced_save_60982
file_prefix(
$savev2_adam_iter_read_readvariableop	*
&savev2_adam_beta_1_read_readvariableop*
&savev2_adam_beta_2_read_readvariableop)
%savev2_adam_decay_read_readvariableop1
-savev2_adam_learning_rate_read_readvariableopE
Asavev2_var_auto_encoder_encoder_conv2d_kernel_read_readvariableopC
?savev2_var_auto_encoder_encoder_conv2d_bias_read_readvariableopG
Csavev2_var_auto_encoder_encoder_conv2d_1_kernel_read_readvariableopE
Asavev2_var_auto_encoder_encoder_conv2d_1_bias_read_readvariableopD
@savev2_var_auto_encoder_encoder_dense_kernel_read_readvariableopB
>savev2_var_auto_encoder_encoder_dense_bias_read_readvariableopF
Bsavev2_var_auto_encoder_encoder_dense_1_kernel_read_readvariableopD
@savev2_var_auto_encoder_encoder_dense_1_bias_read_readvariableopF
Bsavev2_var_auto_encoder_decoder_dense_2_kernel_read_readvariableopD
@savev2_var_auto_encoder_decoder_dense_2_bias_read_readvariableopO
Ksavev2_var_auto_encoder_decoder_conv2d_transpose_kernel_read_readvariableopM
Isavev2_var_auto_encoder_decoder_conv2d_transpose_bias_read_readvariableopQ
Msavev2_var_auto_encoder_decoder_conv2d_transpose_1_kernel_read_readvariableopO
Ksavev2_var_auto_encoder_decoder_conv2d_transpose_1_bias_read_readvariableopQ
Msavev2_var_auto_encoder_decoder_conv2d_transpose_2_kernel_read_readvariableopO
Ksavev2_var_auto_encoder_decoder_conv2d_transpose_2_bias_read_readvariableop$
 savev2_total_read_readvariableop$
 savev2_count_read_readvariableopL
Hsavev2_adam_var_auto_encoder_encoder_conv2d_kernel_m_read_readvariableopJ
Fsavev2_adam_var_auto_encoder_encoder_conv2d_bias_m_read_readvariableopN
Jsavev2_adam_var_auto_encoder_encoder_conv2d_1_kernel_m_read_readvariableopL
Hsavev2_adam_var_auto_encoder_encoder_conv2d_1_bias_m_read_readvariableopK
Gsavev2_adam_var_auto_encoder_encoder_dense_kernel_m_read_readvariableopI
Esavev2_adam_var_auto_encoder_encoder_dense_bias_m_read_readvariableopM
Isavev2_adam_var_auto_encoder_encoder_dense_1_kernel_m_read_readvariableopK
Gsavev2_adam_var_auto_encoder_encoder_dense_1_bias_m_read_readvariableopM
Isavev2_adam_var_auto_encoder_decoder_dense_2_kernel_m_read_readvariableopK
Gsavev2_adam_var_auto_encoder_decoder_dense_2_bias_m_read_readvariableopV
Rsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_kernel_m_read_readvariableopT
Psavev2_adam_var_auto_encoder_decoder_conv2d_transpose_bias_m_read_readvariableopX
Tsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_1_kernel_m_read_readvariableopV
Rsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_1_bias_m_read_readvariableopX
Tsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_2_kernel_m_read_readvariableopV
Rsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_2_bias_m_read_readvariableopL
Hsavev2_adam_var_auto_encoder_encoder_conv2d_kernel_v_read_readvariableopJ
Fsavev2_adam_var_auto_encoder_encoder_conv2d_bias_v_read_readvariableopN
Jsavev2_adam_var_auto_encoder_encoder_conv2d_1_kernel_v_read_readvariableopL
Hsavev2_adam_var_auto_encoder_encoder_conv2d_1_bias_v_read_readvariableopK
Gsavev2_adam_var_auto_encoder_encoder_dense_kernel_v_read_readvariableopI
Esavev2_adam_var_auto_encoder_encoder_dense_bias_v_read_readvariableopM
Isavev2_adam_var_auto_encoder_encoder_dense_1_kernel_v_read_readvariableopK
Gsavev2_adam_var_auto_encoder_encoder_dense_1_bias_v_read_readvariableopM
Isavev2_adam_var_auto_encoder_decoder_dense_2_kernel_v_read_readvariableopK
Gsavev2_adam_var_auto_encoder_decoder_dense_2_bias_v_read_readvariableopV
Rsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_kernel_v_read_readvariableopT
Psavev2_adam_var_auto_encoder_decoder_conv2d_transpose_bias_v_read_readvariableopX
Tsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_1_kernel_v_read_readvariableopV
Rsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_1_bias_v_read_readvariableopX
Tsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_2_kernel_v_read_readvariableopV
Rsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_2_bias_v_read_readvariableop
savev2_const

identity_1??MergeV2Checkpoints?
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1?
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard?
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename?
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
value?B?8B)optimizer/iter/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_1/.ATTRIBUTES/VARIABLE_VALUEB+optimizer/beta_2/.ATTRIBUTES/VARIABLE_VALUEB*optimizer/decay/.ATTRIBUTES/VARIABLE_VALUEB2optimizer/learning_rate/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/0/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/1/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/2/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/3/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/4/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/5/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/6/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/7/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/8/.ATTRIBUTES/VARIABLE_VALUEB0trainable_variables/9/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/10/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/11/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/12/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/13/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/14/.ATTRIBUTES/VARIABLE_VALUEB1trainable_variables/15/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/total/.ATTRIBUTES/VARIABLE_VALUEB4keras_api/metrics/0/count/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/m/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/0/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/1/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/2/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/3/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/4/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/5/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/6/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/7/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/8/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBLtrainable_variables/9/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/10/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/11/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/12/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/13/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/14/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEBMtrainable_variables/15/.OPTIMIZER_SLOT/optimizer/v/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names?
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:8*
dtype0*?
valuezBx8B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B 2
SaveV2/shape_and_slices? 
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0$savev2_adam_iter_read_readvariableop&savev2_adam_beta_1_read_readvariableop&savev2_adam_beta_2_read_readvariableop%savev2_adam_decay_read_readvariableop-savev2_adam_learning_rate_read_readvariableopAsavev2_var_auto_encoder_encoder_conv2d_kernel_read_readvariableop?savev2_var_auto_encoder_encoder_conv2d_bias_read_readvariableopCsavev2_var_auto_encoder_encoder_conv2d_1_kernel_read_readvariableopAsavev2_var_auto_encoder_encoder_conv2d_1_bias_read_readvariableop@savev2_var_auto_encoder_encoder_dense_kernel_read_readvariableop>savev2_var_auto_encoder_encoder_dense_bias_read_readvariableopBsavev2_var_auto_encoder_encoder_dense_1_kernel_read_readvariableop@savev2_var_auto_encoder_encoder_dense_1_bias_read_readvariableopBsavev2_var_auto_encoder_decoder_dense_2_kernel_read_readvariableop@savev2_var_auto_encoder_decoder_dense_2_bias_read_readvariableopKsavev2_var_auto_encoder_decoder_conv2d_transpose_kernel_read_readvariableopIsavev2_var_auto_encoder_decoder_conv2d_transpose_bias_read_readvariableopMsavev2_var_auto_encoder_decoder_conv2d_transpose_1_kernel_read_readvariableopKsavev2_var_auto_encoder_decoder_conv2d_transpose_1_bias_read_readvariableopMsavev2_var_auto_encoder_decoder_conv2d_transpose_2_kernel_read_readvariableopKsavev2_var_auto_encoder_decoder_conv2d_transpose_2_bias_read_readvariableop savev2_total_read_readvariableop savev2_count_read_readvariableopHsavev2_adam_var_auto_encoder_encoder_conv2d_kernel_m_read_readvariableopFsavev2_adam_var_auto_encoder_encoder_conv2d_bias_m_read_readvariableopJsavev2_adam_var_auto_encoder_encoder_conv2d_1_kernel_m_read_readvariableopHsavev2_adam_var_auto_encoder_encoder_conv2d_1_bias_m_read_readvariableopGsavev2_adam_var_auto_encoder_encoder_dense_kernel_m_read_readvariableopEsavev2_adam_var_auto_encoder_encoder_dense_bias_m_read_readvariableopIsavev2_adam_var_auto_encoder_encoder_dense_1_kernel_m_read_readvariableopGsavev2_adam_var_auto_encoder_encoder_dense_1_bias_m_read_readvariableopIsavev2_adam_var_auto_encoder_decoder_dense_2_kernel_m_read_readvariableopGsavev2_adam_var_auto_encoder_decoder_dense_2_bias_m_read_readvariableopRsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_kernel_m_read_readvariableopPsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_bias_m_read_readvariableopTsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_1_kernel_m_read_readvariableopRsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_1_bias_m_read_readvariableopTsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_2_kernel_m_read_readvariableopRsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_2_bias_m_read_readvariableopHsavev2_adam_var_auto_encoder_encoder_conv2d_kernel_v_read_readvariableopFsavev2_adam_var_auto_encoder_encoder_conv2d_bias_v_read_readvariableopJsavev2_adam_var_auto_encoder_encoder_conv2d_1_kernel_v_read_readvariableopHsavev2_adam_var_auto_encoder_encoder_conv2d_1_bias_v_read_readvariableopGsavev2_adam_var_auto_encoder_encoder_dense_kernel_v_read_readvariableopEsavev2_adam_var_auto_encoder_encoder_dense_bias_v_read_readvariableopIsavev2_adam_var_auto_encoder_encoder_dense_1_kernel_v_read_readvariableopGsavev2_adam_var_auto_encoder_encoder_dense_1_bias_v_read_readvariableopIsavev2_adam_var_auto_encoder_decoder_dense_2_kernel_v_read_readvariableopGsavev2_adam_var_auto_encoder_decoder_dense_2_bias_v_read_readvariableopRsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_kernel_v_read_readvariableopPsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_bias_v_read_readvariableopTsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_1_kernel_v_read_readvariableopRsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_1_bias_v_read_readvariableopTsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_2_kernel_v_read_readvariableopRsavev2_adam_var_auto_encoder_decoder_conv2d_transpose_2_bias_v_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *F
dtypes<
:28	2
SaveV2?
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes?
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*?
_input_shapes?
?: : : : : : : : : @:@:	?::	?::	?:?:@ :@: @: : :: : : : : @:@:	?::	?::	?:?:@ :@: @: : :: : : @:@:	?::	?::	?:?:@ :@: @: : :: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 	

_output_shapes
:@:%
!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?:!

_output_shapes	
:?:,(
&
_output_shapes
:@ : 

_output_shapes
:@:,(
&
_output_shapes
: @: 

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
::

_output_shapes
: :

_output_shapes
: :,(
&
_output_shapes
: : 

_output_shapes
: :,(
&
_output_shapes
: @: 

_output_shapes
:@:%!

_output_shapes
:	?: 

_output_shapes
::%!

_output_shapes
:	?: 

_output_shapes
::% !

_output_shapes
:	?:!!

_output_shapes	
:?:,"(
&
_output_shapes
:@ : #

_output_shapes
:@:,$(
&
_output_shapes
: @: %

_output_shapes
: :,&(
&
_output_shapes
: : '

_output_shapes
::,((
&
_output_shapes
: : )

_output_shapes
: :,*(
&
_output_shapes
: @: +

_output_shapes
:@:%,!

_output_shapes
:	?: -

_output_shapes
::%.!

_output_shapes
:	?: /

_output_shapes
::%0!

_output_shapes
:	?:!1

_output_shapes	
:?:,2(
&
_output_shapes
:@ : 3

_output_shapes
:@:,4(
&
_output_shapes
: @: 5

_output_shapes
: :,6(
&
_output_shapes
: : 7

_output_shapes
::8

_output_shapes
: 
?$
?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_59287

inputs,
(conv2d_transpose_readvariableop_resource#
biasadd_readvariableop_resource
identity??BiasAdd/ReadVariableOp?conv2d_transpose/ReadVariableOpD
ShapeShapeinputs*
T0*
_output_shapes
:2
Shapet
strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice/stackx
strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_1x
strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice/stack_2?
strided_sliceStridedSliceShape:output:0strided_slice/stack:output:0strided_slice/stack_1:output:0strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slicex
strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack|
strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_1|
strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_1/stack_2?
strided_slice_1StridedSliceShape:output:0strided_slice_1/stack:output:0 strided_slice_1/stack_1:output:0 strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_1x
strided_slice_2/stackConst*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack|
strided_slice_2/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_1|
strided_slice_2/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_2/stack_2?
strided_slice_2StridedSliceShape:output:0strided_slice_2/stack:output:0 strided_slice_2/stack_1:output:0 strided_slice_2/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_2P
mul/yConst*
_output_shapes
: *
dtype0*
value	B :2
mul/y\
mulMulstrided_slice_1:output:0mul/y:output:0*
T0*
_output_shapes
: 2
mulT
mul_1/yConst*
_output_shapes
: *
dtype0*
value	B :2	
mul_1/yb
mul_1Mulstrided_slice_2:output:0mul_1/y:output:0*
T0*
_output_shapes
: 2
mul_1T
stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2	
stack/3?
stackPackstrided_slice:output:0mul:z:0	mul_1:z:0stack/3:output:0*
N*
T0*
_output_shapes
:2
stackx
strided_slice_3/stackConst*
_output_shapes
:*
dtype0*
valueB: 2
strided_slice_3/stack|
strided_slice_3/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_1|
strided_slice_3/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2
strided_slice_3/stack_2?
strided_slice_3StridedSlicestack:output:0strided_slice_3/stack:output:0 strided_slice_3/stack_1:output:0 strided_slice_3/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
strided_slice_3?
conv2d_transpose/ReadVariableOpReadVariableOp(conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02!
conv2d_transpose/ReadVariableOp?
conv2d_transposeConv2DBackpropInputstack:output:0'conv2d_transpose/ReadVariableOp:value:0inputs*
T0*A
_output_shapes/
-:+??????????????????????????? *
paddingSAME*
strides
2
conv2d_transpose?
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
: *
dtype02
BiasAdd/ReadVariableOp?
BiasAddBiasAddconv2d_transpose:output:0BiasAdd/ReadVariableOp:value:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2	
BiasAddr
ReluReluBiasAdd:output:0*
T0*A
_output_shapes/
-:+??????????????????????????? 2
Relu?
IdentityIdentityRelu:activations:0^BiasAdd/ReadVariableOp ^conv2d_transpose/ReadVariableOp*
T0*A
_output_shapes/
-:+??????????????????????????? 2

Identity"
identityIdentity:output:0*H
_input_shapes7
5:+???????????????????????????@::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2B
conv2d_transpose/ReadVariableOpconv2d_transpose/ReadVariableOp:i e
A
_output_shapes/
-:+???????????????????????????@
 
_user_specified_nameinputs
ϖ
?
 __inference__wrapped_model_59207
input_1B
>var_auto_encoder_encoder_conv2d_conv2d_readvariableop_resourceC
?var_auto_encoder_encoder_conv2d_biasadd_readvariableop_resourceD
@var_auto_encoder_encoder_conv2d_1_conv2d_readvariableop_resourceE
Avar_auto_encoder_encoder_conv2d_1_biasadd_readvariableop_resourceA
=var_auto_encoder_encoder_dense_matmul_readvariableop_resourceB
>var_auto_encoder_encoder_dense_biasadd_readvariableop_resourceC
?var_auto_encoder_encoder_dense_1_matmul_readvariableop_resourceD
@var_auto_encoder_encoder_dense_1_biasadd_readvariableop_resourceC
?var_auto_encoder_decoder_dense_2_matmul_readvariableop_resourceD
@var_auto_encoder_decoder_dense_2_biasadd_readvariableop_resourceV
Rvar_auto_encoder_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resourceM
Ivar_auto_encoder_decoder_conv2d_transpose_biasadd_readvariableop_resourceX
Tvar_auto_encoder_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resourceO
Kvar_auto_encoder_decoder_conv2d_transpose_1_biasadd_readvariableop_resourceX
Tvar_auto_encoder_decoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resourceO
Kvar_auto_encoder_decoder_conv2d_transpose_2_biasadd_readvariableop_resource
identity??@var_auto_encoder/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?Ivar_auto_encoder/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?Bvar_auto_encoder/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?Kvar_auto_encoder/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?Bvar_auto_encoder/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp?Kvar_auto_encoder/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?7var_auto_encoder/decoder/dense_2/BiasAdd/ReadVariableOp?6var_auto_encoder/decoder/dense_2/MatMul/ReadVariableOp?6var_auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp?5var_auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp?8var_auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp?7var_auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp?5var_auto_encoder/encoder/dense/BiasAdd/ReadVariableOp?4var_auto_encoder/encoder/dense/MatMul/ReadVariableOp?7var_auto_encoder/encoder/dense_1/BiasAdd/ReadVariableOp?6var_auto_encoder/encoder/dense_1/MatMul/ReadVariableOp?
5var_auto_encoder/encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp>var_auto_encoder_encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype027
5var_auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp?
&var_auto_encoder/encoder/conv2d/Conv2DConv2Dinput_1=var_auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2(
&var_auto_encoder/encoder/conv2d/Conv2D?
6var_auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp?var_auto_encoder_encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype028
6var_auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp?
'var_auto_encoder/encoder/conv2d/BiasAddBiasAdd/var_auto_encoder/encoder/conv2d/Conv2D:output:0>var_auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2)
'var_auto_encoder/encoder/conv2d/BiasAdd?
$var_auto_encoder/encoder/conv2d/ReluRelu0var_auto_encoder/encoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2&
$var_auto_encoder/encoder/conv2d/Relu?
7var_auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp@var_auto_encoder_encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype029
7var_auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp?
(var_auto_encoder/encoder/conv2d_1/Conv2DConv2D2var_auto_encoder/encoder/conv2d/Relu:activations:0?var_auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2*
(var_auto_encoder/encoder/conv2d_1/Conv2D?
8var_auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOpAvar_auto_encoder_encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02:
8var_auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp?
)var_auto_encoder/encoder/conv2d_1/BiasAddBiasAdd1var_auto_encoder/encoder/conv2d_1/Conv2D:output:0@var_auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2+
)var_auto_encoder/encoder/conv2d_1/BiasAdd?
&var_auto_encoder/encoder/conv2d_1/ReluRelu2var_auto_encoder/encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2(
&var_auto_encoder/encoder/conv2d_1/Relu?
&var_auto_encoder/encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2(
&var_auto_encoder/encoder/flatten/Const?
(var_auto_encoder/encoder/flatten/ReshapeReshape4var_auto_encoder/encoder/conv2d_1/Relu:activations:0/var_auto_encoder/encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2*
(var_auto_encoder/encoder/flatten/Reshape?
4var_auto_encoder/encoder/dense/MatMul/ReadVariableOpReadVariableOp=var_auto_encoder_encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype026
4var_auto_encoder/encoder/dense/MatMul/ReadVariableOp?
%var_auto_encoder/encoder/dense/MatMulMatMul1var_auto_encoder/encoder/flatten/Reshape:output:0<var_auto_encoder/encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2'
%var_auto_encoder/encoder/dense/MatMul?
5var_auto_encoder/encoder/dense/BiasAdd/ReadVariableOpReadVariableOp>var_auto_encoder_encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype027
5var_auto_encoder/encoder/dense/BiasAdd/ReadVariableOp?
&var_auto_encoder/encoder/dense/BiasAddBiasAdd/var_auto_encoder/encoder/dense/MatMul:product:0=var_auto_encoder/encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2(
&var_auto_encoder/encoder/dense/BiasAdd?
6var_auto_encoder/encoder/dense_1/MatMul/ReadVariableOpReadVariableOp?var_auto_encoder_encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype028
6var_auto_encoder/encoder/dense_1/MatMul/ReadVariableOp?
'var_auto_encoder/encoder/dense_1/MatMulMatMul1var_auto_encoder/encoder/flatten/Reshape:output:0>var_auto_encoder/encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2)
'var_auto_encoder/encoder/dense_1/MatMul?
7var_auto_encoder/encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp@var_auto_encoder_encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype029
7var_auto_encoder/encoder/dense_1/BiasAdd/ReadVariableOp?
(var_auto_encoder/encoder/dense_1/BiasAddBiasAdd1var_auto_encoder/encoder/dense_1/MatMul:product:0?var_auto_encoder/encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2*
(var_auto_encoder/encoder/dense_1/BiasAdd?
'var_auto_encoder/encoder/sampling/ShapeShape/var_auto_encoder/encoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:2)
'var_auto_encoder/encoder/sampling/Shape?
4var_auto_encoder/encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    26
4var_auto_encoder/encoder/sampling/random_normal/mean?
6var_auto_encoder/encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??28
6var_auto_encoder/encoder/sampling/random_normal/stddev?
Dvar_auto_encoder/encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormal0var_auto_encoder/encoder/sampling/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype02F
Dvar_auto_encoder/encoder/sampling/random_normal/RandomStandardNormal?
3var_auto_encoder/encoder/sampling/random_normal/mulMulMvar_auto_encoder/encoder/sampling/random_normal/RandomStandardNormal:output:0?var_auto_encoder/encoder/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????25
3var_auto_encoder/encoder/sampling/random_normal/mul?
/var_auto_encoder/encoder/sampling/random_normalAdd7var_auto_encoder/encoder/sampling/random_normal/mul:z:0=var_auto_encoder/encoder/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????21
/var_auto_encoder/encoder/sampling/random_normal?
'var_auto_encoder/encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2)
'var_auto_encoder/encoder/sampling/mul/x?
%var_auto_encoder/encoder/sampling/mulMul0var_auto_encoder/encoder/sampling/mul/x:output:01var_auto_encoder/encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2'
%var_auto_encoder/encoder/sampling/mul?
%var_auto_encoder/encoder/sampling/ExpExp)var_auto_encoder/encoder/sampling/mul:z:0*
T0*'
_output_shapes
:?????????2'
%var_auto_encoder/encoder/sampling/Exp?
'var_auto_encoder/encoder/sampling/mul_1Mul)var_auto_encoder/encoder/sampling/Exp:y:03var_auto_encoder/encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2)
'var_auto_encoder/encoder/sampling/mul_1?
%var_auto_encoder/encoder/sampling/addAddV2/var_auto_encoder/encoder/dense/BiasAdd:output:0+var_auto_encoder/encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2'
%var_auto_encoder/encoder/sampling/add?
6var_auto_encoder/decoder/dense_2/MatMul/ReadVariableOpReadVariableOp?var_auto_encoder_decoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype028
6var_auto_encoder/decoder/dense_2/MatMul/ReadVariableOp?
'var_auto_encoder/decoder/dense_2/MatMulMatMul)var_auto_encoder/encoder/sampling/add:z:0>var_auto_encoder/decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2)
'var_auto_encoder/decoder/dense_2/MatMul?
7var_auto_encoder/decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp@var_auto_encoder_decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype029
7var_auto_encoder/decoder/dense_2/BiasAdd/ReadVariableOp?
(var_auto_encoder/decoder/dense_2/BiasAddBiasAdd1var_auto_encoder/decoder/dense_2/MatMul:product:0?var_auto_encoder/decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2*
(var_auto_encoder/decoder/dense_2/BiasAdd?
%var_auto_encoder/decoder/dense_2/ReluRelu1var_auto_encoder/decoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2'
%var_auto_encoder/decoder/dense_2/Relu?
&var_auto_encoder/decoder/reshape/ShapeShape3var_auto_encoder/decoder/dense_2/Relu:activations:0*
T0*
_output_shapes
:2(
&var_auto_encoder/decoder/reshape/Shape?
4var_auto_encoder/decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 26
4var_auto_encoder/decoder/reshape/strided_slice/stack?
6var_auto_encoder/decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:28
6var_auto_encoder/decoder/reshape/strided_slice/stack_1?
6var_auto_encoder/decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:28
6var_auto_encoder/decoder/reshape/strided_slice/stack_2?
.var_auto_encoder/decoder/reshape/strided_sliceStridedSlice/var_auto_encoder/decoder/reshape/Shape:output:0=var_auto_encoder/decoder/reshape/strided_slice/stack:output:0?var_auto_encoder/decoder/reshape/strided_slice/stack_1:output:0?var_auto_encoder/decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask20
.var_auto_encoder/decoder/reshape/strided_slice?
0var_auto_encoder/decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :22
0var_auto_encoder/decoder/reshape/Reshape/shape/1?
0var_auto_encoder/decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :22
0var_auto_encoder/decoder/reshape/Reshape/shape/2?
0var_auto_encoder/decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 22
0var_auto_encoder/decoder/reshape/Reshape/shape/3?
.var_auto_encoder/decoder/reshape/Reshape/shapePack7var_auto_encoder/decoder/reshape/strided_slice:output:09var_auto_encoder/decoder/reshape/Reshape/shape/1:output:09var_auto_encoder/decoder/reshape/Reshape/shape/2:output:09var_auto_encoder/decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:20
.var_auto_encoder/decoder/reshape/Reshape/shape?
(var_auto_encoder/decoder/reshape/ReshapeReshape3var_auto_encoder/decoder/dense_2/Relu:activations:07var_auto_encoder/decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2*
(var_auto_encoder/decoder/reshape/Reshape?
/var_auto_encoder/decoder/conv2d_transpose/ShapeShape1var_auto_encoder/decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:21
/var_auto_encoder/decoder/conv2d_transpose/Shape?
=var_auto_encoder/decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2?
=var_auto_encoder/decoder/conv2d_transpose/strided_slice/stack?
?var_auto_encoder/decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2A
?var_auto_encoder/decoder/conv2d_transpose/strided_slice/stack_1?
?var_auto_encoder/decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2A
?var_auto_encoder/decoder/conv2d_transpose/strided_slice/stack_2?
7var_auto_encoder/decoder/conv2d_transpose/strided_sliceStridedSlice8var_auto_encoder/decoder/conv2d_transpose/Shape:output:0Fvar_auto_encoder/decoder/conv2d_transpose/strided_slice/stack:output:0Hvar_auto_encoder/decoder/conv2d_transpose/strided_slice/stack_1:output:0Hvar_auto_encoder/decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask29
7var_auto_encoder/decoder/conv2d_transpose/strided_slice?
1var_auto_encoder/decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :23
1var_auto_encoder/decoder/conv2d_transpose/stack/1?
1var_auto_encoder/decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :23
1var_auto_encoder/decoder/conv2d_transpose/stack/2?
1var_auto_encoder/decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@23
1var_auto_encoder/decoder/conv2d_transpose/stack/3?
/var_auto_encoder/decoder/conv2d_transpose/stackPack@var_auto_encoder/decoder/conv2d_transpose/strided_slice:output:0:var_auto_encoder/decoder/conv2d_transpose/stack/1:output:0:var_auto_encoder/decoder/conv2d_transpose/stack/2:output:0:var_auto_encoder/decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:21
/var_auto_encoder/decoder/conv2d_transpose/stack?
?var_auto_encoder/decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?var_auto_encoder/decoder/conv2d_transpose/strided_slice_1/stack?
Avar_auto_encoder/decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Avar_auto_encoder/decoder/conv2d_transpose/strided_slice_1/stack_1?
Avar_auto_encoder/decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Avar_auto_encoder/decoder/conv2d_transpose/strided_slice_1/stack_2?
9var_auto_encoder/decoder/conv2d_transpose/strided_slice_1StridedSlice8var_auto_encoder/decoder/conv2d_transpose/stack:output:0Hvar_auto_encoder/decoder/conv2d_transpose/strided_slice_1/stack:output:0Jvar_auto_encoder/decoder/conv2d_transpose/strided_slice_1/stack_1:output:0Jvar_auto_encoder/decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9var_auto_encoder/decoder/conv2d_transpose/strided_slice_1?
Ivar_auto_encoder/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpRvar_auto_encoder_decoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02K
Ivar_auto_encoder/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?
:var_auto_encoder/decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput8var_auto_encoder/decoder/conv2d_transpose/stack:output:0Qvar_auto_encoder/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:01var_auto_encoder/decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2<
:var_auto_encoder/decoder/conv2d_transpose/conv2d_transpose?
@var_auto_encoder/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOpIvar_auto_encoder_decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02B
@var_auto_encoder/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?
1var_auto_encoder/decoder/conv2d_transpose/BiasAddBiasAddCvar_auto_encoder/decoder/conv2d_transpose/conv2d_transpose:output:0Hvar_auto_encoder/decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@23
1var_auto_encoder/decoder/conv2d_transpose/BiasAdd?
.var_auto_encoder/decoder/conv2d_transpose/ReluRelu:var_auto_encoder/decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@20
.var_auto_encoder/decoder/conv2d_transpose/Relu?
1var_auto_encoder/decoder/conv2d_transpose_1/ShapeShape<var_auto_encoder/decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:23
1var_auto_encoder/decoder/conv2d_transpose_1/Shape?
?var_auto_encoder/decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?var_auto_encoder/decoder/conv2d_transpose_1/strided_slice/stack?
Avar_auto_encoder/decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Avar_auto_encoder/decoder/conv2d_transpose_1/strided_slice/stack_1?
Avar_auto_encoder/decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Avar_auto_encoder/decoder/conv2d_transpose_1/strided_slice/stack_2?
9var_auto_encoder/decoder/conv2d_transpose_1/strided_sliceStridedSlice:var_auto_encoder/decoder/conv2d_transpose_1/Shape:output:0Hvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice/stack:output:0Jvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice/stack_1:output:0Jvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9var_auto_encoder/decoder/conv2d_transpose_1/strided_slice?
3var_auto_encoder/decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :25
3var_auto_encoder/decoder/conv2d_transpose_1/stack/1?
3var_auto_encoder/decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :25
3var_auto_encoder/decoder/conv2d_transpose_1/stack/2?
3var_auto_encoder/decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 25
3var_auto_encoder/decoder/conv2d_transpose_1/stack/3?
1var_auto_encoder/decoder/conv2d_transpose_1/stackPackBvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice:output:0<var_auto_encoder/decoder/conv2d_transpose_1/stack/1:output:0<var_auto_encoder/decoder/conv2d_transpose_1/stack/2:output:0<var_auto_encoder/decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:23
1var_auto_encoder/decoder/conv2d_transpose_1/stack?
Avar_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Avar_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1/stack?
Cvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1/stack_1?
Cvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1/stack_2?
;var_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1StridedSlice:var_auto_encoder/decoder/conv2d_transpose_1/stack:output:0Jvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1/stack:output:0Lvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0Lvar_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;var_auto_encoder/decoder/conv2d_transpose_1/strided_slice_1?
Kvar_auto_encoder/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpTvar_auto_encoder_decoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02M
Kvar_auto_encoder/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
<var_auto_encoder/decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput:var_auto_encoder/decoder/conv2d_transpose_1/stack:output:0Svar_auto_encoder/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0<var_auto_encoder/decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2>
<var_auto_encoder/decoder/conv2d_transpose_1/conv2d_transpose?
Bvar_auto_encoder/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOpKvar_auto_encoder_decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02D
Bvar_auto_encoder/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?
3var_auto_encoder/decoder/conv2d_transpose_1/BiasAddBiasAddEvar_auto_encoder/decoder/conv2d_transpose_1/conv2d_transpose:output:0Jvar_auto_encoder/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 25
3var_auto_encoder/decoder/conv2d_transpose_1/BiasAdd?
0var_auto_encoder/decoder/conv2d_transpose_1/ReluRelu<var_auto_encoder/decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 22
0var_auto_encoder/decoder/conv2d_transpose_1/Relu?
1var_auto_encoder/decoder/conv2d_transpose_2/ShapeShape>var_auto_encoder/decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:23
1var_auto_encoder/decoder/conv2d_transpose_2/Shape?
?var_auto_encoder/decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2A
?var_auto_encoder/decoder/conv2d_transpose_2/strided_slice/stack?
Avar_auto_encoder/decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2C
Avar_auto_encoder/decoder/conv2d_transpose_2/strided_slice/stack_1?
Avar_auto_encoder/decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2C
Avar_auto_encoder/decoder/conv2d_transpose_2/strided_slice/stack_2?
9var_auto_encoder/decoder/conv2d_transpose_2/strided_sliceStridedSlice:var_auto_encoder/decoder/conv2d_transpose_2/Shape:output:0Hvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice/stack:output:0Jvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice/stack_1:output:0Jvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2;
9var_auto_encoder/decoder/conv2d_transpose_2/strided_slice?
3var_auto_encoder/decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :25
3var_auto_encoder/decoder/conv2d_transpose_2/stack/1?
3var_auto_encoder/decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :25
3var_auto_encoder/decoder/conv2d_transpose_2/stack/2?
3var_auto_encoder/decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :25
3var_auto_encoder/decoder/conv2d_transpose_2/stack/3?
1var_auto_encoder/decoder/conv2d_transpose_2/stackPackBvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice:output:0<var_auto_encoder/decoder/conv2d_transpose_2/stack/1:output:0<var_auto_encoder/decoder/conv2d_transpose_2/stack/2:output:0<var_auto_encoder/decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:23
1var_auto_encoder/decoder/conv2d_transpose_2/stack?
Avar_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 2C
Avar_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1/stack?
Cvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2E
Cvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1/stack_1?
Cvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2E
Cvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1/stack_2?
;var_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1StridedSlice:var_auto_encoder/decoder/conv2d_transpose_2/stack:output:0Jvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1/stack:output:0Lvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0Lvar_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2=
;var_auto_encoder/decoder/conv2d_transpose_2/strided_slice_1?
Kvar_auto_encoder/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpTvar_auto_encoder_decoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02M
Kvar_auto_encoder/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
<var_auto_encoder/decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput:var_auto_encoder/decoder/conv2d_transpose_2/stack:output:0Svar_auto_encoder/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0>var_auto_encoder/decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2>
<var_auto_encoder/decoder/conv2d_transpose_2/conv2d_transpose?
Bvar_auto_encoder/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOpKvar_auto_encoder_decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02D
Bvar_auto_encoder/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp?
3var_auto_encoder/decoder/conv2d_transpose_2/BiasAddBiasAddEvar_auto_encoder/decoder/conv2d_transpose_2/conv2d_transpose:output:0Jvar_auto_encoder/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????25
3var_auto_encoder/decoder/conv2d_transpose_2/BiasAdd?
)var_auto_encoder/logistic_loss/zeros_like	ZerosLike<var_auto_encoder/decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2+
)var_auto_encoder/logistic_loss/zeros_like?
+var_auto_encoder/logistic_loss/GreaterEqualGreaterEqual<var_auto_encoder/decoder/conv2d_transpose_2/BiasAdd:output:0-var_auto_encoder/logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2-
+var_auto_encoder/logistic_loss/GreaterEqual?
%var_auto_encoder/logistic_loss/SelectSelect/var_auto_encoder/logistic_loss/GreaterEqual:z:0<var_auto_encoder/decoder/conv2d_transpose_2/BiasAdd:output:0-var_auto_encoder/logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2'
%var_auto_encoder/logistic_loss/Select?
"var_auto_encoder/logistic_loss/NegNeg<var_auto_encoder/decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2$
"var_auto_encoder/logistic_loss/Neg?
'var_auto_encoder/logistic_loss/Select_1Select/var_auto_encoder/logistic_loss/GreaterEqual:z:0&var_auto_encoder/logistic_loss/Neg:y:0<var_auto_encoder/decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2)
'var_auto_encoder/logistic_loss/Select_1?
"var_auto_encoder/logistic_loss/mulMul<var_auto_encoder/decoder/conv2d_transpose_2/BiasAdd:output:0input_1*
T0*/
_output_shapes
:?????????2$
"var_auto_encoder/logistic_loss/mul?
"var_auto_encoder/logistic_loss/subSub.var_auto_encoder/logistic_loss/Select:output:0&var_auto_encoder/logistic_loss/mul:z:0*
T0*/
_output_shapes
:?????????2$
"var_auto_encoder/logistic_loss/sub?
"var_auto_encoder/logistic_loss/ExpExp0var_auto_encoder/logistic_loss/Select_1:output:0*
T0*/
_output_shapes
:?????????2$
"var_auto_encoder/logistic_loss/Exp?
$var_auto_encoder/logistic_loss/Log1pLog1p&var_auto_encoder/logistic_loss/Exp:y:0*
T0*/
_output_shapes
:?????????2&
$var_auto_encoder/logistic_loss/Log1p?
var_auto_encoder/logistic_lossAdd&var_auto_encoder/logistic_loss/sub:z:0(var_auto_encoder/logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????2 
var_auto_encoder/logistic_loss?
&var_auto_encoder/Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2(
&var_auto_encoder/Sum/reduction_indices?
var_auto_encoder/SumSum"var_auto_encoder/logistic_loss:z:0/var_auto_encoder/Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
var_auto_encoder/Sum?
var_auto_encoder/NegNegvar_auto_encoder/Sum:output:0*
T0*#
_output_shapes
:?????????2
var_auto_encoder/Neg?
 var_auto_encoder/PartitionedCallPartitionedCall)var_auto_encoder/encoder/sampling/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591782"
 var_auto_encoder/PartitionedCall?
"var_auto_encoder/PartitionedCall_1PartitionedCall)var_auto_encoder/encoder/sampling/add:z:0/var_auto_encoder/encoder/dense/BiasAdd:output:01var_auto_encoder/encoder/dense_1/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591992$
"var_auto_encoder/PartitionedCall_1?
var_auto_encoder/subSub)var_auto_encoder/PartitionedCall:output:0+var_auto_encoder/PartitionedCall_1:output:0*
T0*#
_output_shapes
:?????????2
var_auto_encoder/sub?
var_auto_encoder/addAddV2var_auto_encoder/Neg:y:0var_auto_encoder/sub:z:0*
T0*#
_output_shapes
:?????????2
var_auto_encoder/addz
var_auto_encoder/ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
var_auto_encoder/Const?
var_auto_encoder/MeanMeanvar_auto_encoder/add:z:0var_auto_encoder/Const:output:0*
T0*
_output_shapes
: 2
var_auto_encoder/Meanx
var_auto_encoder/Neg_1Negvar_auto_encoder/Mean:output:0*
T0*
_output_shapes
: 2
var_auto_encoder/Neg_1?	
IdentityIdentity<var_auto_encoder/decoder/conv2d_transpose_2/BiasAdd:output:0A^var_auto_encoder/decoder/conv2d_transpose/BiasAdd/ReadVariableOpJ^var_auto_encoder/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpC^var_auto_encoder/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpL^var_auto_encoder/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpC^var_auto_encoder/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpL^var_auto_encoder/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp8^var_auto_encoder/decoder/dense_2/BiasAdd/ReadVariableOp7^var_auto_encoder/decoder/dense_2/MatMul/ReadVariableOp7^var_auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp6^var_auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp9^var_auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp8^var_auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp6^var_auto_encoder/encoder/dense/BiasAdd/ReadVariableOp5^var_auto_encoder/encoder/dense/MatMul/ReadVariableOp8^var_auto_encoder/encoder/dense_1/BiasAdd/ReadVariableOp7^var_auto_encoder/encoder/dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::2?
@var_auto_encoder/decoder/conv2d_transpose/BiasAdd/ReadVariableOp@var_auto_encoder/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2?
Ivar_auto_encoder/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpIvar_auto_encoder/decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2?
Bvar_auto_encoder/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpBvar_auto_encoder/decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2?
Kvar_auto_encoder/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpKvar_auto_encoder/decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2?
Bvar_auto_encoder/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpBvar_auto_encoder/decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2?
Kvar_auto_encoder/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpKvar_auto_encoder/decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2r
7var_auto_encoder/decoder/dense_2/BiasAdd/ReadVariableOp7var_auto_encoder/decoder/dense_2/BiasAdd/ReadVariableOp2p
6var_auto_encoder/decoder/dense_2/MatMul/ReadVariableOp6var_auto_encoder/decoder/dense_2/MatMul/ReadVariableOp2p
6var_auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp6var_auto_encoder/encoder/conv2d/BiasAdd/ReadVariableOp2n
5var_auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp5var_auto_encoder/encoder/conv2d/Conv2D/ReadVariableOp2t
8var_auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp8var_auto_encoder/encoder/conv2d_1/BiasAdd/ReadVariableOp2r
7var_auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp7var_auto_encoder/encoder/conv2d_1/Conv2D/ReadVariableOp2n
5var_auto_encoder/encoder/dense/BiasAdd/ReadVariableOp5var_auto_encoder/encoder/dense/BiasAdd/ReadVariableOp2l
4var_auto_encoder/encoder/dense/MatMul/ReadVariableOp4var_auto_encoder/encoder/dense/MatMul/ReadVariableOp2r
7var_auto_encoder/encoder/dense_1/BiasAdd/ReadVariableOp7var_auto_encoder/encoder/dense_1/BiasAdd/ReadVariableOp2p
6var_auto_encoder/encoder/dense_1/MatMul/ReadVariableOp6var_auto_encoder/encoder/dense_1/MatMul/ReadVariableOp:X T
/
_output_shapes
:?????????
!
_user_specified_name	input_1
?

?
0__inference_var_auto_encoder_layer_call_fn_60412

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
	unknown_7
	unknown_8
	unknown_9

unknown_10

unknown_11

unknown_12

unknown_13

unknown_14
identity??StatefulPartitionedCall?
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6	unknown_7	unknown_8	unknown_9
unknown_10
unknown_11
unknown_12
unknown_13
unknown_14*
Tin
2*
Tout
2*
_collective_manager_ids
 *1
_output_shapes
:?????????: *2
_read_only_resource_inputs
	
*0
config_proto 

CPU

GPU2*0J 8? *T
fORM
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_599732
StatefulPartitionedCall?
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity"
identityIdentity:output:0*n
_input_shapes]
[:?????????::::::::::::::::22
StatefulPartitionedCallStatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_60196

inputs1
-encoder_conv2d_conv2d_readvariableop_resource2
.encoder_conv2d_biasadd_readvariableop_resource3
/encoder_conv2d_1_conv2d_readvariableop_resource4
0encoder_conv2d_1_biasadd_readvariableop_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource2
.encoder_dense_1_matmul_readvariableop_resource3
/encoder_dense_1_biasadd_readvariableop_resource2
.decoder_dense_2_matmul_readvariableop_resource3
/decoder_dense_2_biasadd_readvariableop_resourceE
Adecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource<
8decoder_conv2d_transpose_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_1_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_2_biasadd_readvariableop_resource
identity

identity_1??/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?&decoder/dense_2/BiasAdd/ReadVariableOp?%decoder/dense_2/MatMul/ReadVariableOp?%encoder/conv2d/BiasAdd/ReadVariableOp?$encoder/conv2d/Conv2D/ReadVariableOp?'encoder/conv2d_1/BiasAdd/ReadVariableOp?&encoder/conv2d_1/Conv2D/ReadVariableOp?$encoder/dense/BiasAdd/ReadVariableOp?#encoder/dense/MatMul/ReadVariableOp?&encoder/dense_1/BiasAdd/ReadVariableOp?%encoder/dense_1/MatMul/ReadVariableOp?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOp?
encoder/conv2d/Conv2DConv2Dinputs,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
encoder/conv2d/Conv2D?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOp?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d/BiasAdd?
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d/Relu?
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02(
&encoder/conv2d_1/Conv2D/ReadVariableOp?
encoder/conv2d_1/Conv2DConv2D!encoder/conv2d/Relu:activations:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
encoder/conv2d_1/Conv2D?
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_1/BiasAdd/ReadVariableOp?
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_1/BiasAdd?
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_1/Relu
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
encoder/flatten/Const?
encoder/flatten/ReshapeReshape#encoder/conv2d_1/Relu:activations:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten/Reshape?
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#encoder/dense/MatMul/ReadVariableOp?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense/MatMul?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense/BiasAdd?
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%encoder/dense_1/MatMul/ReadVariableOp?
encoder/dense_1/MatMulMatMul encoder/flatten/Reshape:output:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/MatMul?
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_1/BiasAdd/ReadVariableOp?
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/BiasAdd~
encoder/sampling/ShapeShapeencoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling/Shape?
#encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#encoder/sampling/random_normal/mean?
%encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%encoder/sampling/random_normal/stddev?
3encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormalencoder/sampling/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype025
3encoder/sampling/random_normal/RandomStandardNormal?
"encoder/sampling/random_normal/mulMul<encoder/sampling/random_normal/RandomStandardNormal:output:0.encoder/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2$
"encoder/sampling/random_normal/mul?
encoder/sampling/random_normalAdd&encoder/sampling/random_normal/mul:z:0,encoder/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2 
encoder/sampling/random_normalu
encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
encoder/sampling/mul/x?
encoder/sampling/mulMulencoder/sampling/mul/x:output:0 encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
encoder/sampling/mul
encoder/sampling/ExpExpencoder/sampling/mul:z:0*
T0*'
_output_shapes
:?????????2
encoder/sampling/Exp?
encoder/sampling/mul_1Mulencoder/sampling/Exp:y:0"encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
encoder/sampling/mul_1?
encoder/sampling/addAddV2encoder/dense/BiasAdd:output:0encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
encoder/sampling/add?
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%decoder/dense_2/MatMul/ReadVariableOp?
decoder/dense_2/MatMulMatMulencoder/sampling/add:z:0-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_2/MatMul?
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_2/BiasAdd/ReadVariableOp?
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_2/BiasAdd?
decoder/dense_2/ReluRelu decoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
decoder/dense_2/Relu?
decoder/reshape/ShapeShape"decoder/dense_2/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape/Shape?
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack?
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1?
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice?
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/1?
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/2?
decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2!
decoder/reshape/Reshape/shape/3?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0(decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape?
decoder/reshape/ReshapeReshape"decoder/dense_2/Relu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
decoder/reshape/Reshape?
decoder/conv2d_transpose/ShapeShape decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:2 
decoder/conv2d_transpose/Shape?
,decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder/conv2d_transpose/strided_slice/stack?
.decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/conv2d_transpose/strided_slice/stack_1?
.decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/conv2d_transpose/strided_slice/stack_2?
&decoder/conv2d_transpose/strided_sliceStridedSlice'decoder/conv2d_transpose/Shape:output:05decoder/conv2d_transpose/strided_slice/stack:output:07decoder/conv2d_transpose/strided_slice/stack_1:output:07decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder/conv2d_transpose/strided_slice?
 decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose/stack/1?
 decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose/stack/2?
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2"
 decoder/conv2d_transpose/stack/3?
decoder/conv2d_transpose/stackPack/decoder/conv2d_transpose/strided_slice:output:0)decoder/conv2d_transpose/stack/1:output:0)decoder/conv2d_transpose/stack/2:output:0)decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2 
decoder/conv2d_transpose/stack?
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose/strided_slice_1/stack?
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose/strided_slice_1/stack_1?
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose/strided_slice_1/stack_2?
(decoder/conv2d_transpose/strided_slice_1StridedSlice'decoder/conv2d_transpose/stack:output:07decoder/conv2d_transpose/strided_slice_1/stack:output:09decoder/conv2d_transpose/strided_slice_1/stack_1:output:09decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose/strided_slice_1?
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02:
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput'decoder/conv2d_transpose/stack:output:0@decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0 decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2+
)decoder/conv2d_transpose/conv2d_transpose?
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?
 decoder/conv2d_transpose/BiasAddBiasAdd2decoder/conv2d_transpose/conv2d_transpose:output:07decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2"
 decoder/conv2d_transpose/BiasAdd?
decoder/conv2d_transpose/ReluRelu)decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
decoder/conv2d_transpose/Relu?
 decoder/conv2d_transpose_1/ShapeShape+decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_1/Shape?
.decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_1/strided_slice/stack?
0decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_1/strided_slice/stack_1?
0decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_1/strided_slice/stack_2?
(decoder/conv2d_transpose_1/strided_sliceStridedSlice)decoder/conv2d_transpose_1/Shape:output:07decoder/conv2d_transpose_1/strided_slice/stack:output:09decoder/conv2d_transpose_1/strided_slice/stack_1:output:09decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_1/strided_slice?
"decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_1/stack/1?
"decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_1/stack/2?
"decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"decoder/conv2d_transpose_1/stack/3?
 decoder/conv2d_transpose_1/stackPack1decoder/conv2d_transpose_1/strided_slice:output:0+decoder/conv2d_transpose_1/stack/1:output:0+decoder/conv2d_transpose_1/stack/2:output:0+decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_1/stack?
0decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_1/strided_slice_1/stack?
2decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_1/strided_slice_1/stack_1?
2decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_1/strided_slice_1/stack_2?
*decoder/conv2d_transpose_1/strided_slice_1StridedSlice)decoder/conv2d_transpose_1/stack:output:09decoder/conv2d_transpose_1/strided_slice_1/stack:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_1/strided_slice_1?
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_1/stack:output:0Bdecoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2-
+decoder/conv2d_transpose_1/conv2d_transpose?
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_1/BiasAddBiasAdd4decoder/conv2d_transpose_1/conv2d_transpose:output:09decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2$
"decoder/conv2d_transpose_1/BiasAdd?
decoder/conv2d_transpose_1/ReluRelu+decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2!
decoder/conv2d_transpose_1/Relu?
 decoder/conv2d_transpose_2/ShapeShape-decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_2/Shape?
.decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_2/strided_slice/stack?
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_2/strided_slice/stack_1?
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_2/strided_slice/stack_2?
(decoder/conv2d_transpose_2/strided_sliceStridedSlice)decoder/conv2d_transpose_2/Shape:output:07decoder/conv2d_transpose_2/strided_slice/stack:output:09decoder/conv2d_transpose_2/strided_slice/stack_1:output:09decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_2/strided_slice?
"decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_2/stack/1?
"decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_2/stack/2?
"decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_2/stack/3?
 decoder/conv2d_transpose_2/stackPack1decoder/conv2d_transpose_2/strided_slice:output:0+decoder/conv2d_transpose_2/stack/1:output:0+decoder/conv2d_transpose_2/stack/2:output:0+decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_2/stack?
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_2/strided_slice_1/stack?
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_2/strided_slice_1/stack_1?
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_2/strided_slice_1/stack_2?
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice)decoder/conv2d_transpose_2/stack:output:09decoder/conv2d_transpose_2/strided_slice_1/stack:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_2/strided_slice_1?
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02<
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_2/stack:output:0Bdecoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_2/conv2d_transpose?
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_2/BiasAddBiasAdd4decoder/conv2d_transpose_2/conv2d_transpose:output:09decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"decoder/conv2d_transpose_2/BiasAdd?
logistic_loss/zeros_like	ZerosLike+decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/zeros_like?
logistic_loss/GreaterEqualGreaterEqual+decoder/conv2d_transpose_2/BiasAdd:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/GreaterEqual?
logistic_loss/SelectSelectlogistic_loss/GreaterEqual:z:0+decoder/conv2d_transpose_2/BiasAdd:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select?
logistic_loss/NegNeg+decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Neg?
logistic_loss/Select_1Selectlogistic_loss/GreaterEqual:z:0logistic_loss/Neg:y:0+decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select_1?
logistic_loss/mulMul+decoder/conv2d_transpose_2/BiasAdd:output:0inputs*
T0*/
_output_shapes
:?????????2
logistic_loss/mul?
logistic_loss/subSublogistic_loss/Select:output:0logistic_loss/mul:z:0*
T0*/
_output_shapes
:?????????2
logistic_loss/sub?
logistic_loss/ExpExplogistic_loss/Select_1:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Exp?
logistic_loss/Log1pLog1plogistic_loss/Exp:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Log1p?
logistic_lossAddlogistic_loss/sub:z:0logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss?
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Sum/reduction_indicesr
SumSumlogistic_loss:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumM
NegNegSum:output:0*
T0*#
_output_shapes
:?????????2
Neg?
PartitionedCallPartitionedCallencoder/sampling/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591782
PartitionedCall?
PartitionedCall_1PartitionedCallencoder/sampling/add:z:0encoder/dense/BiasAdd:output:0 encoder/dense_1/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591992
PartitionedCall_1u
subSubPartitionedCall:output:0PartitionedCall_1:output:0*
T0*#
_output_shapes
:?????????2
subS
addAddV2Neg:y:0sub:z:0*
T0*#
_output_shapes
:?????????2
addX
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstN
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: 2
MeanE
Neg_1NegMean:output:0*
T0*
_output_shapes
: 2
Neg_1?
IdentityIdentity+decoder/conv2d_transpose_2/BiasAdd:output:00^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp'^decoder/dense_2/BiasAdd/ReadVariableOp&^decoder/dense_2/MatMul/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity	Neg_1:y:00^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp'^decoder/dense_2/BiasAdd/ReadVariableOp&^decoder/dense_2/MatMul/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2b
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2t
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2P
&decoder/dense_2/BiasAdd/ReadVariableOp&decoder/dense_2/BiasAdd/ReadVariableOp2N
%decoder/dense_2/MatMul/ReadVariableOp%decoder/dense_2/MatMul/ReadVariableOp2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp2P
&encoder/dense_1/BiasAdd/ReadVariableOp&encoder/dense_1/BiasAdd/ReadVariableOp2N
%encoder/dense_1/MatMul/ReadVariableOp%encoder/dense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
??
?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_60336

inputs1
-encoder_conv2d_conv2d_readvariableop_resource2
.encoder_conv2d_biasadd_readvariableop_resource3
/encoder_conv2d_1_conv2d_readvariableop_resource4
0encoder_conv2d_1_biasadd_readvariableop_resource0
,encoder_dense_matmul_readvariableop_resource1
-encoder_dense_biasadd_readvariableop_resource2
.encoder_dense_1_matmul_readvariableop_resource3
/encoder_dense_1_biasadd_readvariableop_resource2
.decoder_dense_2_matmul_readvariableop_resource3
/decoder_dense_2_biasadd_readvariableop_resourceE
Adecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource<
8decoder_conv2d_transpose_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_1_biasadd_readvariableop_resourceG
Cdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource>
:decoder_conv2d_transpose_2_biasadd_readvariableop_resource
identity

identity_1??/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp?:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?&decoder/dense_2/BiasAdd/ReadVariableOp?%decoder/dense_2/MatMul/ReadVariableOp?%encoder/conv2d/BiasAdd/ReadVariableOp?$encoder/conv2d/Conv2D/ReadVariableOp?'encoder/conv2d_1/BiasAdd/ReadVariableOp?&encoder/conv2d_1/Conv2D/ReadVariableOp?$encoder/dense/BiasAdd/ReadVariableOp?#encoder/dense/MatMul/ReadVariableOp?&encoder/dense_1/BiasAdd/ReadVariableOp?%encoder/dense_1/MatMul/ReadVariableOp?
$encoder/conv2d/Conv2D/ReadVariableOpReadVariableOp-encoder_conv2d_conv2d_readvariableop_resource*&
_output_shapes
: *
dtype02&
$encoder/conv2d/Conv2D/ReadVariableOp?
encoder/conv2d/Conv2DConv2Dinputs,encoder/conv2d/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? *
paddingVALID*
strides
2
encoder/conv2d/Conv2D?
%encoder/conv2d/BiasAdd/ReadVariableOpReadVariableOp.encoder_conv2d_biasadd_readvariableop_resource*
_output_shapes
: *
dtype02'
%encoder/conv2d/BiasAdd/ReadVariableOp?
encoder/conv2d/BiasAddBiasAddencoder/conv2d/Conv2D:output:0-encoder/conv2d/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d/BiasAdd?
encoder/conv2d/ReluReluencoder/conv2d/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2
encoder/conv2d/Relu?
&encoder/conv2d_1/Conv2D/ReadVariableOpReadVariableOp/encoder_conv2d_1_conv2d_readvariableop_resource*&
_output_shapes
: @*
dtype02(
&encoder/conv2d_1/Conv2D/ReadVariableOp?
encoder/conv2d_1/Conv2DConv2D!encoder/conv2d/Relu:activations:0.encoder/conv2d_1/Conv2D/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@*
paddingVALID*
strides
2
encoder/conv2d_1/Conv2D?
'encoder/conv2d_1/BiasAdd/ReadVariableOpReadVariableOp0encoder_conv2d_1_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype02)
'encoder/conv2d_1/BiasAdd/ReadVariableOp?
encoder/conv2d_1/BiasAddBiasAdd encoder/conv2d_1/Conv2D:output:0/encoder/conv2d_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_1/BiasAdd?
encoder/conv2d_1/ReluRelu!encoder/conv2d_1/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
encoder/conv2d_1/Relu
encoder/flatten/ConstConst*
_output_shapes
:*
dtype0*
valueB"???? 	  2
encoder/flatten/Const?
encoder/flatten/ReshapeReshape#encoder/conv2d_1/Relu:activations:0encoder/flatten/Const:output:0*
T0*(
_output_shapes
:??????????2
encoder/flatten/Reshape?
#encoder/dense/MatMul/ReadVariableOpReadVariableOp,encoder_dense_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02%
#encoder/dense/MatMul/ReadVariableOp?
encoder/dense/MatMulMatMul encoder/flatten/Reshape:output:0+encoder/dense/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense/MatMul?
$encoder/dense/BiasAdd/ReadVariableOpReadVariableOp-encoder_dense_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02&
$encoder/dense/BiasAdd/ReadVariableOp?
encoder/dense/BiasAddBiasAddencoder/dense/MatMul:product:0,encoder/dense/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense/BiasAdd?
%encoder/dense_1/MatMul/ReadVariableOpReadVariableOp.encoder_dense_1_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%encoder/dense_1/MatMul/ReadVariableOp?
encoder/dense_1/MatMulMatMul encoder/flatten/Reshape:output:0-encoder/dense_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/MatMul?
&encoder/dense_1/BiasAdd/ReadVariableOpReadVariableOp/encoder_dense_1_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/dense_1/BiasAdd/ReadVariableOp?
encoder/dense_1/BiasAddBiasAdd encoder/dense_1/MatMul:product:0.encoder/dense_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:?????????2
encoder/dense_1/BiasAdd~
encoder/sampling/ShapeShapeencoder/dense/BiasAdd:output:0*
T0*
_output_shapes
:2
encoder/sampling/Shape?
#encoder/sampling/random_normal/meanConst*
_output_shapes
: *
dtype0*
valueB
 *    2%
#encoder/sampling/random_normal/mean?
%encoder/sampling/random_normal/stddevConst*
_output_shapes
: *
dtype0*
valueB
 *  ??2'
%encoder/sampling/random_normal/stddev?
3encoder/sampling/random_normal/RandomStandardNormalRandomStandardNormalencoder/sampling/Shape:output:0*
T0*'
_output_shapes
:?????????*
dtype025
3encoder/sampling/random_normal/RandomStandardNormal?
"encoder/sampling/random_normal/mulMul<encoder/sampling/random_normal/RandomStandardNormal:output:0.encoder/sampling/random_normal/stddev:output:0*
T0*'
_output_shapes
:?????????2$
"encoder/sampling/random_normal/mul?
encoder/sampling/random_normalAdd&encoder/sampling/random_normal/mul:z:0,encoder/sampling/random_normal/mean:output:0*
T0*'
_output_shapes
:?????????2 
encoder/sampling/random_normalu
encoder/sampling/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *   ?2
encoder/sampling/mul/x?
encoder/sampling/mulMulencoder/sampling/mul/x:output:0 encoder/dense_1/BiasAdd:output:0*
T0*'
_output_shapes
:?????????2
encoder/sampling/mul
encoder/sampling/ExpExpencoder/sampling/mul:z:0*
T0*'
_output_shapes
:?????????2
encoder/sampling/Exp?
encoder/sampling/mul_1Mulencoder/sampling/Exp:y:0"encoder/sampling/random_normal:z:0*
T0*'
_output_shapes
:?????????2
encoder/sampling/mul_1?
encoder/sampling/addAddV2encoder/dense/BiasAdd:output:0encoder/sampling/mul_1:z:0*
T0*'
_output_shapes
:?????????2
encoder/sampling/add?
%decoder/dense_2/MatMul/ReadVariableOpReadVariableOp.decoder_dense_2_matmul_readvariableop_resource*
_output_shapes
:	?*
dtype02'
%decoder/dense_2/MatMul/ReadVariableOp?
decoder/dense_2/MatMulMatMulencoder/sampling/add:z:0-decoder/dense_2/MatMul/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_2/MatMul?
&decoder/dense_2/BiasAdd/ReadVariableOpReadVariableOp/decoder_dense_2_biasadd_readvariableop_resource*
_output_shapes	
:?*
dtype02(
&decoder/dense_2/BiasAdd/ReadVariableOp?
decoder/dense_2/BiasAddBiasAdd decoder/dense_2/MatMul:product:0.decoder/dense_2/BiasAdd/ReadVariableOp:value:0*
T0*(
_output_shapes
:??????????2
decoder/dense_2/BiasAdd?
decoder/dense_2/ReluRelu decoder/dense_2/BiasAdd:output:0*
T0*(
_output_shapes
:??????????2
decoder/dense_2/Relu?
decoder/reshape/ShapeShape"decoder/dense_2/Relu:activations:0*
T0*
_output_shapes
:2
decoder/reshape/Shape?
#decoder/reshape/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2%
#decoder/reshape/strided_slice/stack?
%decoder/reshape/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_1?
%decoder/reshape/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:2'
%decoder/reshape/strided_slice/stack_2?
decoder/reshape/strided_sliceStridedSlicedecoder/reshape/Shape:output:0,decoder/reshape/strided_slice/stack:output:0.decoder/reshape/strided_slice/stack_1:output:0.decoder/reshape/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2
decoder/reshape/strided_slice?
decoder/reshape/Reshape/shape/1Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/1?
decoder/reshape/Reshape/shape/2Const*
_output_shapes
: *
dtype0*
value	B :2!
decoder/reshape/Reshape/shape/2?
decoder/reshape/Reshape/shape/3Const*
_output_shapes
: *
dtype0*
value	B : 2!
decoder/reshape/Reshape/shape/3?
decoder/reshape/Reshape/shapePack&decoder/reshape/strided_slice:output:0(decoder/reshape/Reshape/shape/1:output:0(decoder/reshape/Reshape/shape/2:output:0(decoder/reshape/Reshape/shape/3:output:0*
N*
T0*
_output_shapes
:2
decoder/reshape/Reshape/shape?
decoder/reshape/ReshapeReshape"decoder/dense_2/Relu:activations:0&decoder/reshape/Reshape/shape:output:0*
T0*/
_output_shapes
:????????? 2
decoder/reshape/Reshape?
decoder/conv2d_transpose/ShapeShape decoder/reshape/Reshape:output:0*
T0*
_output_shapes
:2 
decoder/conv2d_transpose/Shape?
,decoder/conv2d_transpose/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 2.
,decoder/conv2d_transpose/strided_slice/stack?
.decoder/conv2d_transpose/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/conv2d_transpose/strided_slice/stack_1?
.decoder/conv2d_transpose/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:20
.decoder/conv2d_transpose/strided_slice/stack_2?
&decoder/conv2d_transpose/strided_sliceStridedSlice'decoder/conv2d_transpose/Shape:output:05decoder/conv2d_transpose/strided_slice/stack:output:07decoder/conv2d_transpose/strided_slice/stack_1:output:07decoder/conv2d_transpose/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2(
&decoder/conv2d_transpose/strided_slice?
 decoder/conv2d_transpose/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose/stack/1?
 decoder/conv2d_transpose/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2"
 decoder/conv2d_transpose/stack/2?
 decoder/conv2d_transpose/stack/3Const*
_output_shapes
: *
dtype0*
value	B :@2"
 decoder/conv2d_transpose/stack/3?
decoder/conv2d_transpose/stackPack/decoder/conv2d_transpose/strided_slice:output:0)decoder/conv2d_transpose/stack/1:output:0)decoder/conv2d_transpose/stack/2:output:0)decoder/conv2d_transpose/stack/3:output:0*
N*
T0*
_output_shapes
:2 
decoder/conv2d_transpose/stack?
.decoder/conv2d_transpose/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose/strided_slice_1/stack?
0decoder/conv2d_transpose/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose/strided_slice_1/stack_1?
0decoder/conv2d_transpose/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose/strided_slice_1/stack_2?
(decoder/conv2d_transpose/strided_slice_1StridedSlice'decoder/conv2d_transpose/stack:output:07decoder/conv2d_transpose/strided_slice_1/stack:output:09decoder/conv2d_transpose/strided_slice_1/stack_1:output:09decoder/conv2d_transpose/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose/strided_slice_1?
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOpReadVariableOpAdecoder_conv2d_transpose_conv2d_transpose_readvariableop_resource*&
_output_shapes
:@ *
dtype02:
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp?
)decoder/conv2d_transpose/conv2d_transposeConv2DBackpropInput'decoder/conv2d_transpose/stack:output:0@decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp:value:0 decoder/reshape/Reshape:output:0*
T0*/
_output_shapes
:?????????@*
paddingSAME*
strides
2+
)decoder/conv2d_transpose/conv2d_transpose?
/decoder/conv2d_transpose/BiasAdd/ReadVariableOpReadVariableOp8decoder_conv2d_transpose_biasadd_readvariableop_resource*
_output_shapes
:@*
dtype021
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp?
 decoder/conv2d_transpose/BiasAddBiasAdd2decoder/conv2d_transpose/conv2d_transpose:output:07decoder/conv2d_transpose/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????@2"
 decoder/conv2d_transpose/BiasAdd?
decoder/conv2d_transpose/ReluRelu)decoder/conv2d_transpose/BiasAdd:output:0*
T0*/
_output_shapes
:?????????@2
decoder/conv2d_transpose/Relu?
 decoder/conv2d_transpose_1/ShapeShape+decoder/conv2d_transpose/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_1/Shape?
.decoder/conv2d_transpose_1/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_1/strided_slice/stack?
0decoder/conv2d_transpose_1/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_1/strided_slice/stack_1?
0decoder/conv2d_transpose_1/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_1/strided_slice/stack_2?
(decoder/conv2d_transpose_1/strided_sliceStridedSlice)decoder/conv2d_transpose_1/Shape:output:07decoder/conv2d_transpose_1/strided_slice/stack:output:09decoder/conv2d_transpose_1/strided_slice/stack_1:output:09decoder/conv2d_transpose_1/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_1/strided_slice?
"decoder/conv2d_transpose_1/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_1/stack/1?
"decoder/conv2d_transpose_1/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_1/stack/2?
"decoder/conv2d_transpose_1/stack/3Const*
_output_shapes
: *
dtype0*
value	B : 2$
"decoder/conv2d_transpose_1/stack/3?
 decoder/conv2d_transpose_1/stackPack1decoder/conv2d_transpose_1/strided_slice:output:0+decoder/conv2d_transpose_1/stack/1:output:0+decoder/conv2d_transpose_1/stack/2:output:0+decoder/conv2d_transpose_1/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_1/stack?
0decoder/conv2d_transpose_1/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_1/strided_slice_1/stack?
2decoder/conv2d_transpose_1/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_1/strided_slice_1/stack_1?
2decoder/conv2d_transpose_1/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_1/strided_slice_1/stack_2?
*decoder/conv2d_transpose_1/strided_slice_1StridedSlice)decoder/conv2d_transpose_1/stack:output:09decoder/conv2d_transpose_1/strided_slice_1/stack:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_1/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_1/strided_slice_1?
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_1_conv2d_transpose_readvariableop_resource*&
_output_shapes
: @*
dtype02<
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_1/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_1/stack:output:0Bdecoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:value:0+decoder/conv2d_transpose/Relu:activations:0*
T0*/
_output_shapes
:????????? *
paddingSAME*
strides
2-
+decoder/conv2d_transpose_1/conv2d_transpose?
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_1_biasadd_readvariableop_resource*
_output_shapes
: *
dtype023
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_1/BiasAddBiasAdd4decoder/conv2d_transpose_1/conv2d_transpose:output:09decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:????????? 2$
"decoder/conv2d_transpose_1/BiasAdd?
decoder/conv2d_transpose_1/ReluRelu+decoder/conv2d_transpose_1/BiasAdd:output:0*
T0*/
_output_shapes
:????????? 2!
decoder/conv2d_transpose_1/Relu?
 decoder/conv2d_transpose_2/ShapeShape-decoder/conv2d_transpose_1/Relu:activations:0*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_2/Shape?
.decoder/conv2d_transpose_2/strided_slice/stackConst*
_output_shapes
:*
dtype0*
valueB: 20
.decoder/conv2d_transpose_2/strided_slice/stack?
0decoder/conv2d_transpose_2/strided_slice/stack_1Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_2/strided_slice/stack_1?
0decoder/conv2d_transpose_2/strided_slice/stack_2Const*
_output_shapes
:*
dtype0*
valueB:22
0decoder/conv2d_transpose_2/strided_slice/stack_2?
(decoder/conv2d_transpose_2/strided_sliceStridedSlice)decoder/conv2d_transpose_2/Shape:output:07decoder/conv2d_transpose_2/strided_slice/stack:output:09decoder/conv2d_transpose_2/strided_slice/stack_1:output:09decoder/conv2d_transpose_2/strided_slice/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2*
(decoder/conv2d_transpose_2/strided_slice?
"decoder/conv2d_transpose_2/stack/1Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_2/stack/1?
"decoder/conv2d_transpose_2/stack/2Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_2/stack/2?
"decoder/conv2d_transpose_2/stack/3Const*
_output_shapes
: *
dtype0*
value	B :2$
"decoder/conv2d_transpose_2/stack/3?
 decoder/conv2d_transpose_2/stackPack1decoder/conv2d_transpose_2/strided_slice:output:0+decoder/conv2d_transpose_2/stack/1:output:0+decoder/conv2d_transpose_2/stack/2:output:0+decoder/conv2d_transpose_2/stack/3:output:0*
N*
T0*
_output_shapes
:2"
 decoder/conv2d_transpose_2/stack?
0decoder/conv2d_transpose_2/strided_slice_1/stackConst*
_output_shapes
:*
dtype0*
valueB: 22
0decoder/conv2d_transpose_2/strided_slice_1/stack?
2decoder/conv2d_transpose_2/strided_slice_1/stack_1Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_2/strided_slice_1/stack_1?
2decoder/conv2d_transpose_2/strided_slice_1/stack_2Const*
_output_shapes
:*
dtype0*
valueB:24
2decoder/conv2d_transpose_2/strided_slice_1/stack_2?
*decoder/conv2d_transpose_2/strided_slice_1StridedSlice)decoder/conv2d_transpose_2/stack:output:09decoder/conv2d_transpose_2/strided_slice_1/stack:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_1:output:0;decoder/conv2d_transpose_2/strided_slice_1/stack_2:output:0*
Index0*
T0*
_output_shapes
: *
shrink_axis_mask2,
*decoder/conv2d_transpose_2/strided_slice_1?
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOpReadVariableOpCdecoder_conv2d_transpose_2_conv2d_transpose_readvariableop_resource*&
_output_shapes
: *
dtype02<
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp?
+decoder/conv2d_transpose_2/conv2d_transposeConv2DBackpropInput)decoder/conv2d_transpose_2/stack:output:0Bdecoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:value:0-decoder/conv2d_transpose_1/Relu:activations:0*
T0*/
_output_shapes
:?????????*
paddingSAME*
strides
2-
+decoder/conv2d_transpose_2/conv2d_transpose?
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOpReadVariableOp:decoder_conv2d_transpose_2_biasadd_readvariableop_resource*
_output_shapes
:*
dtype023
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp?
"decoder/conv2d_transpose_2/BiasAddBiasAdd4decoder/conv2d_transpose_2/conv2d_transpose:output:09decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp:value:0*
T0*/
_output_shapes
:?????????2$
"decoder/conv2d_transpose_2/BiasAdd?
logistic_loss/zeros_like	ZerosLike+decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/zeros_like?
logistic_loss/GreaterEqualGreaterEqual+decoder/conv2d_transpose_2/BiasAdd:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/GreaterEqual?
logistic_loss/SelectSelectlogistic_loss/GreaterEqual:z:0+decoder/conv2d_transpose_2/BiasAdd:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select?
logistic_loss/NegNeg+decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Neg?
logistic_loss/Select_1Selectlogistic_loss/GreaterEqual:z:0logistic_loss/Neg:y:0+decoder/conv2d_transpose_2/BiasAdd:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select_1?
logistic_loss/mulMul+decoder/conv2d_transpose_2/BiasAdd:output:0inputs*
T0*/
_output_shapes
:?????????2
logistic_loss/mul?
logistic_loss/subSublogistic_loss/Select:output:0logistic_loss/mul:z:0*
T0*/
_output_shapes
:?????????2
logistic_loss/sub?
logistic_loss/ExpExplogistic_loss/Select_1:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Exp?
logistic_loss/Log1pLog1plogistic_loss/Exp:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Log1p?
logistic_lossAddlogistic_loss/sub:z:0logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss?
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Sum/reduction_indicesr
SumSumlogistic_loss:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumM
NegNegSum:output:0*
T0*#
_output_shapes
:?????????2
Neg?
PartitionedCallPartitionedCallencoder/sampling/add:z:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591782
PartitionedCall?
PartitionedCall_1PartitionedCallencoder/sampling/add:z:0encoder/dense/BiasAdd:output:0 encoder/dense_1/BiasAdd:output:0*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591992
PartitionedCall_1u
subSubPartitionedCall:output:0PartitionedCall_1:output:0*
T0*#
_output_shapes
:?????????2
subS
addAddV2Neg:y:0sub:z:0*
T0*#
_output_shapes
:?????????2
addX
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstN
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: 2
MeanE
Neg_1NegMean:output:0*
T0*
_output_shapes
: 2
Neg_1?
IdentityIdentity+decoder/conv2d_transpose_2/BiasAdd:output:00^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp'^decoder/dense_2/BiasAdd/ReadVariableOp&^decoder/dense_2/MatMul/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity	Neg_1:y:00^decoder/conv2d_transpose/BiasAdd/ReadVariableOp9^decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2^decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp;^decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp'^decoder/dense_2/BiasAdd/ReadVariableOp&^decoder/dense_2/MatMul/ReadVariableOp&^encoder/conv2d/BiasAdd/ReadVariableOp%^encoder/conv2d/Conv2D/ReadVariableOp(^encoder/conv2d_1/BiasAdd/ReadVariableOp'^encoder/conv2d_1/Conv2D/ReadVariableOp%^encoder/dense/BiasAdd/ReadVariableOp$^encoder/dense/MatMul/ReadVariableOp'^encoder/dense_1/BiasAdd/ReadVariableOp&^encoder/dense_1/MatMul/ReadVariableOp*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2b
/decoder/conv2d_transpose/BiasAdd/ReadVariableOp/decoder/conv2d_transpose/BiasAdd/ReadVariableOp2t
8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp8decoder/conv2d_transpose/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_1/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_1/conv2d_transpose/ReadVariableOp2f
1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp1decoder/conv2d_transpose_2/BiasAdd/ReadVariableOp2x
:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp:decoder/conv2d_transpose_2/conv2d_transpose/ReadVariableOp2P
&decoder/dense_2/BiasAdd/ReadVariableOp&decoder/dense_2/BiasAdd/ReadVariableOp2N
%decoder/dense_2/MatMul/ReadVariableOp%decoder/dense_2/MatMul/ReadVariableOp2N
%encoder/conv2d/BiasAdd/ReadVariableOp%encoder/conv2d/BiasAdd/ReadVariableOp2L
$encoder/conv2d/Conv2D/ReadVariableOp$encoder/conv2d/Conv2D/ReadVariableOp2R
'encoder/conv2d_1/BiasAdd/ReadVariableOp'encoder/conv2d_1/BiasAdd/ReadVariableOp2P
&encoder/conv2d_1/Conv2D/ReadVariableOp&encoder/conv2d_1/Conv2D/ReadVariableOp2L
$encoder/dense/BiasAdd/ReadVariableOp$encoder/dense/BiasAdd/ReadVariableOp2J
#encoder/dense/MatMul/ReadVariableOp#encoder/dense/MatMul/ReadVariableOp2P
&encoder/dense_1/BiasAdd/ReadVariableOp&encoder/dense_1/BiasAdd/ReadVariableOp2N
%encoder/dense_1/MatMul/ReadVariableOp%encoder/dense_1/MatMul/ReadVariableOp:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs
?+
?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_59973

inputs
encoder_59915
encoder_59917
encoder_59919
encoder_59921
encoder_59923
encoder_59925
encoder_59927
encoder_59929
decoder_59934
decoder_59936
decoder_59938
decoder_59940
decoder_59942
decoder_59944
decoder_59946
decoder_59948
identity

identity_1??decoder/StatefulPartitionedCall?encoder/StatefulPartitionedCall?
encoder/StatefulPartitionedCallStatefulPartitionedCallinputsencoder_59915encoder_59917encoder_59919encoder_59921encoder_59923encoder_59925encoder_59927encoder_59929*
Tin
2	*
Tout
2*
_collective_manager_ids
 *M
_output_shapes;
9:?????????:?????????:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_encoder_layer_call_and_return_conditional_losses_594352!
encoder/StatefulPartitionedCall?
decoder/StatefulPartitionedCallStatefulPartitionedCall(encoder/StatefulPartitionedCall:output:2decoder_59934decoder_59936decoder_59938decoder_59940decoder_59942decoder_59944decoder_59946decoder_59948*
Tin
2	*
Tout
2*
_collective_manager_ids
 */
_output_shapes
:?????????**
_read_only_resource_inputs

*0
config_proto 

CPU

GPU2*0J 8? *K
fFRD
B__inference_decoder_layer_call_and_return_conditional_losses_596662!
decoder/StatefulPartitionedCall?
logistic_loss/zeros_like	ZerosLike(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/zeros_like?
logistic_loss/GreaterEqualGreaterEqual(decoder/StatefulPartitionedCall:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/GreaterEqual?
logistic_loss/SelectSelectlogistic_loss/GreaterEqual:z:0(decoder/StatefulPartitionedCall:output:0logistic_loss/zeros_like:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select?
logistic_loss/NegNeg(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Neg?
logistic_loss/Select_1Selectlogistic_loss/GreaterEqual:z:0logistic_loss/Neg:y:0(decoder/StatefulPartitionedCall:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Select_1?
logistic_loss/mulMul(decoder/StatefulPartitionedCall:output:0inputs*
T0*/
_output_shapes
:?????????2
logistic_loss/mul?
logistic_loss/subSublogistic_loss/Select:output:0logistic_loss/mul:z:0*
T0*/
_output_shapes
:?????????2
logistic_loss/sub?
logistic_loss/ExpExplogistic_loss/Select_1:output:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Exp?
logistic_loss/Log1pLog1plogistic_loss/Exp:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss/Log1p?
logistic_lossAddlogistic_loss/sub:z:0logistic_loss/Log1p:y:0*
T0*/
_output_shapes
:?????????2
logistic_loss?
Sum/reduction_indicesConst*
_output_shapes
:*
dtype0*!
valueB"         2
Sum/reduction_indicesr
SumSumlogistic_loss:z:0Sum/reduction_indices:output:0*
T0*#
_output_shapes
:?????????2
SumM
NegNegSum:output:0*
T0*#
_output_shapes
:?????????2
Neg?
PartitionedCallPartitionedCall(encoder/StatefulPartitionedCall:output:2*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591782
PartitionedCall?
PartitionedCall_1PartitionedCall(encoder/StatefulPartitionedCall:output:2(encoder/StatefulPartitionedCall:output:0(encoder/StatefulPartitionedCall:output:1*
Tin
2*
Tout
2*
_collective_manager_ids
 *#
_output_shapes
:?????????* 
_read_only_resource_inputs
 *0
config_proto 

CPU

GPU2*0J 8? *)
f$R"
 __inference_log_normal_pdf_591992
PartitionedCall_1u
subSubPartitionedCall:output:0PartitionedCall_1:output:0*
T0*#
_output_shapes
:?????????2
subS
addAddV2Neg:y:0sub:z:0*
T0*#
_output_shapes
:?????????2
addX
ConstConst*
_output_shapes
:*
dtype0*
valueB: 2
ConstN
MeanMeanadd:z:0Const:output:0*
T0*
_output_shapes
: 2
MeanE
Neg_1NegMean:output:0*
T0*
_output_shapes
: 2
Neg_1?
IdentityIdentity(decoder/StatefulPartitionedCall:output:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*/
_output_shapes
:?????????2

Identity?

Identity_1Identity	Neg_1:y:0 ^decoder/StatefulPartitionedCall ^encoder/StatefulPartitionedCall*
T0*
_output_shapes
: 2

Identity_1"
identityIdentity:output:0"!

identity_1Identity_1:output:0*n
_input_shapes]
[:?????????::::::::::::::::2B
decoder/StatefulPartitionedCalldecoder/StatefulPartitionedCall2B
encoder/StatefulPartitionedCallencoder/StatefulPartitionedCall:W S
/
_output_shapes
:?????????
 
_user_specified_nameinputs"?L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*?
serving_default?
C
input_18
serving_default_input_1:0?????????D
output_18
StatefulPartitionedCall:0?????????tensorflow/serving/predict:??
?
encoder
decoder
	optimizer
loss
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
?__call__
+?&call_and_return_all_conditional_losses
?_default_save_signature
?log_normal_pdf"?
_tf_keras_model?{"class_name": "VarAutoEncoder", "name": "var_auto_encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"layer was saved without config": true}, "is_graph_network": false, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "VarAutoEncoder"}, "training_config": {"loss": null, "metrics": null, "weighted_metrics": null, "loss_weights": null, "optimizer_config": {"class_name": "Adam", "config": {"name": "Adam", "learning_rate": 9.999999747378752e-05, "decay": 0.0, "beta_1": 0.8999999761581421, "beta_2": 0.9990000128746033, "epsilon": 1e-07, "amsgrad": false}}}}
?
	
conv1
	conv2
flatten
dense3_1
dense3_2
sampling
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Encoder", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"layer was saved without config": true}}
?

dense1
reshape
deconv1
deconv2
out
trainable_variables
regularization_losses
	variables
	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Decoder", "name": "decoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "decoder", "trainable": true, "dtype": "float32"}}
?
iter

beta_1

beta_2
	 decay
!learning_rate"m?#m?$m?%m?&m?'m?(m?)m?*m?+m?,m?-m?.m?/m?0m?1m?"v?#v?$v?%v?&v?'v?(v?)v?*v?+v?,v?-v?.v?/v?0v?1v?"
	optimizer
 "
trackable_dict_wrapper
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115"
trackable_list_wrapper
 "
trackable_list_wrapper
?
"0
#1
$2
%3
&4
'5
(6
)7
*8
+9
,10
-11
.12
/13
014
115"
trackable_list_wrapper
?

2layers
3non_trainable_variables
trainable_variables
4layer_regularization_losses
regularization_losses
5metrics
6layer_metrics
	variables
?__call__
?_default_save_signature
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
-
?serving_default"
signature_map
?	

"kernel
#bias
7trainable_variables
8regularization_losses
9	variables
:	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 1}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 1]}}
?	

$kernel
%bias
;trainable_variables
<regularization_losses
=	variables
>	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2D", "name": "conv2d_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_1", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "valid", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 4, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 13, 13, 32]}}
?
?trainable_variables
@regularization_losses
A	variables
B	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Flatten", "name": "flatten", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "flatten", "trainable": true, "dtype": "float32", "data_format": "channels_last"}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 1, "axes": {}}}}
?

&kernel
'bias
Ctrainable_variables
Dregularization_losses
E	variables
F	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
?

(kernel
)bias
Gtrainable_variables
Hregularization_losses
I	variables
J	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_1", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2304}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2304]}}
?
Ktrainable_variables
Lregularization_losses
M	variables
N	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Sampling", "name": "sampling", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "sampling", "trainable": true, "dtype": "float32"}}
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
 "
trackable_list_wrapper
X
"0
#1
$2
%3
&4
'5
(6
)7"
trackable_list_wrapper
?

Olayers
Pnon_trainable_variables
trainable_variables
Qlayer_regularization_losses
regularization_losses
Rmetrics
Slayer_metrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
?

*kernel
+bias
Ttrainable_variables
Uregularization_losses
V	variables
W	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Dense", "name": "dense_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "dense_2", "trainable": true, "dtype": "float32", "units": 1568, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 2}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 2]}}
?
Xtrainable_variables
Yregularization_losses
Z	variables
[	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Reshape", "name": "reshape", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "reshape", "trainable": true, "dtype": "float32", "target_shape": {"class_name": "__tuple__", "items": [7, 7, 32]}}}
?


,kernel
-bias
\trainable_variables
]regularization_losses
^	variables
_	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose", "trainable": true, "dtype": "float32", "filters": 64, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 7, 7, 32]}}
?


.kernel
/bias
`trainable_variables
aregularization_losses
b	variables
c	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_1", "trainable": true, "dtype": "float32", "filters": 32, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [2, 2]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "relu", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 64}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 14, 14, 64]}}
?


0kernel
1bias
dtrainable_variables
eregularization_losses
f	variables
g	keras_api
?__call__
+?&call_and_return_all_conditional_losses"?	
_tf_keras_layer?{"class_name": "Conv2DTranspose", "name": "conv2d_transpose_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "conv2d_transpose_2", "trainable": true, "dtype": "float32", "filters": 1, "kernel_size": {"class_name": "__tuple__", "items": [3, 3]}, "strides": {"class_name": "__tuple__", "items": [1, 1]}, "padding": "same", "data_format": "channels_last", "dilation_rate": {"class_name": "__tuple__", "items": [1, 1]}, "groups": 1, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "GlorotUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "Zeros", "config": {}}, "kernel_regularizer": null, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null, "output_padding": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": 4, "max_ndim": null, "min_ndim": null, "axes": {"-1": 32}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 28, 28, 32]}}
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
 "
trackable_list_wrapper
X
*0
+1
,2
-3
.4
/5
06
17"
trackable_list_wrapper
?

hlayers
inon_trainable_variables
trainable_variables
jlayer_regularization_losses
regularization_losses
kmetrics
llayer_metrics
	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
:	 (2	Adam/iter
: (2Adam/beta_1
: (2Adam/beta_2
: (2
Adam/decay
: (2Adam/learning_rate
@:> 2&var_auto_encoder/encoder/conv2d/kernel
2:0 2$var_auto_encoder/encoder/conv2d/bias
B:@ @2(var_auto_encoder/encoder/conv2d_1/kernel
4:2@2&var_auto_encoder/encoder/conv2d_1/bias
8:6	?2%var_auto_encoder/encoder/dense/kernel
1:/2#var_auto_encoder/encoder/dense/bias
::8	?2'var_auto_encoder/encoder/dense_1/kernel
3:12%var_auto_encoder/encoder/dense_1/bias
::8	?2'var_auto_encoder/decoder/dense_2/kernel
4:2?2%var_auto_encoder/decoder/dense_2/bias
J:H@ 20var_auto_encoder/decoder/conv2d_transpose/kernel
<::@2.var_auto_encoder/decoder/conv2d_transpose/bias
L:J @22var_auto_encoder/decoder/conv2d_transpose_1/kernel
>:< 20var_auto_encoder/decoder/conv2d_transpose_1/bias
L:J 22var_auto_encoder/decoder/conv2d_transpose_2/kernel
>:<20var_auto_encoder/decoder/conv2d_transpose_2/bias
.
0
1"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
'
m0"
trackable_list_wrapper
 "
trackable_dict_wrapper
.
"0
#1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
"0
#1"
trackable_list_wrapper
?

nlayers
onon_trainable_variables
7trainable_variables
player_regularization_losses
8regularization_losses
qmetrics
rlayer_metrics
9	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
$0
%1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
$0
%1"
trackable_list_wrapper
?

slayers
tnon_trainable_variables
;trainable_variables
ulayer_regularization_losses
<regularization_losses
vmetrics
wlayer_metrics
=	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?

xlayers
ynon_trainable_variables
?trainable_variables
zlayer_regularization_losses
@regularization_losses
{metrics
|layer_metrics
A	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
&0
'1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
&0
'1"
trackable_list_wrapper
?

}layers
~non_trainable_variables
Ctrainable_variables
layer_regularization_losses
Dregularization_losses
?metrics
?layer_metrics
E	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
(0
)1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
(0
)1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
Gtrainable_variables
 ?layer_regularization_losses
Hregularization_losses
?metrics
?layer_metrics
I	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
Ktrainable_variables
 ?layer_regularization_losses
Lregularization_losses
?metrics
?layer_metrics
M	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
J

0
1
2
3
4
5"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
.
*0
+1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
*0
+1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
Ttrainable_variables
 ?layer_regularization_losses
Uregularization_losses
?metrics
?layer_metrics
V	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
?
?layers
?non_trainable_variables
Xtrainable_variables
 ?layer_regularization_losses
Yregularization_losses
?metrics
?layer_metrics
Z	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
,0
-1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
,0
-1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
\trainable_variables
 ?layer_regularization_losses
]regularization_losses
?metrics
?layer_metrics
^	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
.0
/1"
trackable_list_wrapper
 "
trackable_list_wrapper
.
.0
/1"
trackable_list_wrapper
?
?layers
?non_trainable_variables
`trainable_variables
 ?layer_regularization_losses
aregularization_losses
?metrics
?layer_metrics
b	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
.
00
11"
trackable_list_wrapper
 "
trackable_list_wrapper
.
00
11"
trackable_list_wrapper
?
?layers
?non_trainable_variables
dtrainable_variables
 ?layer_regularization_losses
eregularization_losses
?metrics
?layer_metrics
f	variables
?__call__
+?&call_and_return_all_conditional_losses
'?"call_and_return_conditional_losses"
_generic_user_object
C
0
1
2
3
4"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
?

?total

?count
?	variables
?	keras_api"?
_tf_keras_metricj{"class_name": "Mean", "name": "loss", "dtype": "float32", "config": {"name": "loss", "dtype": "float32"}}
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
:  (2total
:  (2count
0
?0
?1"
trackable_list_wrapper
.
?	variables"
_generic_user_object
E:C 2-Adam/var_auto_encoder/encoder/conv2d/kernel/m
7:5 2+Adam/var_auto_encoder/encoder/conv2d/bias/m
G:E @2/Adam/var_auto_encoder/encoder/conv2d_1/kernel/m
9:7@2-Adam/var_auto_encoder/encoder/conv2d_1/bias/m
=:;	?2,Adam/var_auto_encoder/encoder/dense/kernel/m
6:42*Adam/var_auto_encoder/encoder/dense/bias/m
?:=	?2.Adam/var_auto_encoder/encoder/dense_1/kernel/m
8:62,Adam/var_auto_encoder/encoder/dense_1/bias/m
?:=	?2.Adam/var_auto_encoder/decoder/dense_2/kernel/m
9:7?2,Adam/var_auto_encoder/decoder/dense_2/bias/m
O:M@ 27Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/m
A:?@25Adam/var_auto_encoder/decoder/conv2d_transpose/bias/m
Q:O @29Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/m
C:A 27Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/m
Q:O 29Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/m
C:A27Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/m
E:C 2-Adam/var_auto_encoder/encoder/conv2d/kernel/v
7:5 2+Adam/var_auto_encoder/encoder/conv2d/bias/v
G:E @2/Adam/var_auto_encoder/encoder/conv2d_1/kernel/v
9:7@2-Adam/var_auto_encoder/encoder/conv2d_1/bias/v
=:;	?2,Adam/var_auto_encoder/encoder/dense/kernel/v
6:42*Adam/var_auto_encoder/encoder/dense/bias/v
?:=	?2.Adam/var_auto_encoder/encoder/dense_1/kernel/v
8:62,Adam/var_auto_encoder/encoder/dense_1/bias/v
?:=	?2.Adam/var_auto_encoder/decoder/dense_2/kernel/v
9:7?2,Adam/var_auto_encoder/decoder/dense_2/bias/v
O:M@ 27Adam/var_auto_encoder/decoder/conv2d_transpose/kernel/v
A:?@25Adam/var_auto_encoder/decoder/conv2d_transpose/bias/v
Q:O @29Adam/var_auto_encoder/decoder/conv2d_transpose_1/kernel/v
C:A 27Adam/var_auto_encoder/decoder/conv2d_transpose_1/bias/v
Q:O 29Adam/var_auto_encoder/decoder/conv2d_transpose_2/kernel/v
C:A27Adam/var_auto_encoder/decoder/conv2d_transpose_2/bias/v
?2?
0__inference_var_auto_encoder_layer_call_fn_60412
0__inference_var_auto_encoder_layer_call_fn_60374
0__inference_var_auto_encoder_layer_call_fn_60009
0__inference_var_auto_encoder_layer_call_fn_59910?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_60336
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_60196
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_59810
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_59749?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
 __inference__wrapped_model_59207?
???
FullArgSpec
args? 
varargsjargs
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *.?+
)?&
input_1?????????
?2?
 __inference_log_normal_pdf_60432
 __inference_log_normal_pdf_60452?
???
FullArgSpec*
args"?
jself
jz
jmean
jlogvar
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
'__inference_encoder_layer_call_fn_60592
'__inference_encoder_layer_call_fn_60567?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
B__inference_encoder_layer_call_and_return_conditional_losses_60542
B__inference_encoder_layer_call_and_return_conditional_losses_60497?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
'__inference_decoder_layer_call_fn_60773
'__inference_decoder_layer_call_fn_60794?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2?
B__inference_decoder_layer_call_and_return_conditional_losses_60752
B__inference_decoder_layer_call_and_return_conditional_losses_60672?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?B?
#__inference_signature_wrapper_60056input_1"?
???
FullArgSpec
args? 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkwjkwargs
defaults? 

kwonlyargs?

jtraining%
kwonlydefaults?

trainingp 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2??
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *
 
?2?
0__inference_conv2d_transpose_layer_call_fn_59252?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_59242?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
2__inference_conv2d_transpose_1_layer_call_fn_59297?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_59287?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+???????????????????????????@
?2?
2__inference_conv2d_transpose_2_layer_call_fn_59341?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? 
?2?
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_59331?
???
FullArgSpec
args?
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs? 
kwonlydefaults
 
annotations? *7?4
2?/+??????????????????????????? ?
 __inference__wrapped_model_59207?"#$%&'()*+,-./018?5
.?+
)?&
input_1?????????
? ";?8
6
output_1*?'
output_1??????????
M__inference_conv2d_transpose_1_layer_call_and_return_conditional_losses_59287?./I?F
??<
:?7
inputs+???????????????????????????@
? "??<
5?2
0+??????????????????????????? 
? ?
2__inference_conv2d_transpose_1_layer_call_fn_59297?./I?F
??<
:?7
inputs+???????????????????????????@
? "2?/+??????????????????????????? ?
M__inference_conv2d_transpose_2_layer_call_and_return_conditional_losses_59331?01I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????
? ?
2__inference_conv2d_transpose_2_layer_call_fn_59341?01I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+????????????????????????????
K__inference_conv2d_transpose_layer_call_and_return_conditional_losses_59242?,-I?F
??<
:?7
inputs+??????????????????????????? 
? "??<
5?2
0+???????????????????????????@
? ?
0__inference_conv2d_transpose_layer_call_fn_59252?,-I?F
??<
:?7
inputs+??????????????????????????? 
? "2?/+???????????????????????????@?
B__inference_decoder_layer_call_and_return_conditional_losses_60672z*+,-./01??<
%?"
 ?
inputs?????????
?

trainingp"-?*
#? 
0?????????
? ?
B__inference_decoder_layer_call_and_return_conditional_losses_60752z*+,-./01??<
%?"
 ?
inputs?????????
?

trainingp "-?*
#? 
0?????????
? ?
'__inference_decoder_layer_call_fn_60773m*+,-./01??<
%?"
 ?
inputs?????????
?

trainingp" ???????????
'__inference_decoder_layer_call_fn_60794m*+,-./01??<
%?"
 ?
inputs?????????
?

trainingp " ???????????
B__inference_encoder_layer_call_and_return_conditional_losses_60497?"#$%&'()G?D
-?*
(?%
inputs?????????
?

trainingp"j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
B__inference_encoder_layer_call_and_return_conditional_losses_60542?"#$%&'()G?D
-?*
(?%
inputs?????????
?

trainingp "j?g
`?]
?
0/0?????????
?
0/1?????????
?
0/2?????????
? ?
'__inference_encoder_layer_call_fn_60567?"#$%&'()G?D
-?*
(?%
inputs?????????
?

trainingp"Z?W
?
0?????????
?
1?????????
?
2??????????
'__inference_encoder_layer_call_fn_60592?"#$%&'()G?D
-?*
(?%
inputs?????????
?

trainingp "Z?W
?
0?????????
?
1?????????
?
2?????????|
 __inference_log_normal_pdf_60432X@?=
6?3
?
z?????????
	Y        
	Y        
? "???????????
 __inference_log_normal_pdf_60452?l?i
b?_
?
z?????????
?
mean?????????
 ?
logvar?????????
? "???????????
#__inference_signature_wrapper_60056?"#$%&'()*+,-./01C?@
? 
9?6
4
input_1)?&
input_1?????????";?8
6
output_1*?'
output_1??????????
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_59749?"#$%&'()*+,-./01H?E
.?+
)?&
input_1?????????
?

trainingp";?8
#? 
0?????????
?
?	
1/0 ?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_59810?"#$%&'()*+,-./01H?E
.?+
)?&
input_1?????????
?

trainingp ";?8
#? 
0?????????
?
?	
1/0 ?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_60196?"#$%&'()*+,-./01G?D
-?*
(?%
inputs?????????
?

trainingp";?8
#? 
0?????????
?
?	
1/0 ?
K__inference_var_auto_encoder_layer_call_and_return_conditional_losses_60336?"#$%&'()*+,-./01G?D
-?*
(?%
inputs?????????
?

trainingp ";?8
#? 
0?????????
?
?	
1/0 ?
0__inference_var_auto_encoder_layer_call_fn_59910~"#$%&'()*+,-./01H?E
.?+
)?&
input_1?????????
?

trainingp" ???????????
0__inference_var_auto_encoder_layer_call_fn_60009~"#$%&'()*+,-./01H?E
.?+
)?&
input_1?????????
?

trainingp " ???????????
0__inference_var_auto_encoder_layer_call_fn_60374}"#$%&'()*+,-./01G?D
-?*
(?%
inputs?????????
?

trainingp" ???????????
0__inference_var_auto_encoder_layer_call_fn_60412}"#$%&'()*+,-./01G?D
-?*
(?%
inputs?????????
?

trainingp " ??????????