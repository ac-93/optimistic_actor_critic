̪
�)�)
.
Abs
x"T
y"T"
Ttype:

2	
:
Add
x"T
y"T
z"T"
Ttype:
2	
W
AddN
inputs"T*N
sum"T"
Nint(0"!
Ttype:
2	��
A
AddV2
x"T
y"T
z"T"
Ttype:
2	��
�
	ApplyAdam
var"T�	
m"T�	
v"T�
beta1_power"T
beta2_power"T
lr"T

beta1"T

beta2"T
epsilon"T	
grad"T
out"T�" 
Ttype:
2	"
use_lockingbool( "
use_nesterovbool( 
�
ArgMax

input"T
	dimension"Tidx
output"output_type" 
Ttype:
2	"
Tidxtype0:
2	"
output_typetype0	:
2	
x
Assign
ref"T�

value"T

output_ref"T�"	
Ttype"
validate_shapebool("
use_lockingbool(�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
~
BiasAddGrad
out_backprop"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
R
BroadcastGradientArgs
s0"T
s1"T
r0"T
r1"T"
Ttype0:
2	
N
Cast	
x"SrcT	
y"DstT"
SrcTtype"
DstTtype"
Truncatebool( 
h
ConcatV2
values"T*N
axis"Tidx
output"T"
Nint(0"	
Ttype"
Tidxtype0:
2	
8
Const
output"dtype"
valuetensor"
dtypetype
S
DynamicStitch
indices*N
data"T*N
merged"T"
Nint(0"	
Ttype
,
Exp
x"T
y"T"
Ttype:

2
^
Fill
dims"
index_type

value"T
output"T"	
Ttype"

index_typetype0:
2	
?
FloorDiv
x"T
y"T
z"T"
Ttype:
2	
9
FloorMod
x"T
y"T
z"T"
Ttype:

2	
=
Greater
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
?

LogSoftmax
logits"T

logsoftmax"T"
Ttype:
2
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
8
Maximum
x"T
y"T
z"T"
Ttype:

2	
�
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
delete_old_dirsbool(�
8
Minimum
x"T
y"T
z"T"
Ttype:

2	
=
Mul
x"T
y"T
z"T"
Ttype:
2	�
�
Multinomial
logits"T
num_samples
output"output_dtype"
seedint "
seed2int "
Ttype:
2	" 
output_dtypetype0	:
2	�
.
Neg
x"T
y"T"
Ttype:

2	

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
X
PlaceholderWithDefault
input"dtype
output"dtype"
dtypetype"
shapeshape
6
Pow
x"T
y"T
z"T"
Ttype:

2	
�
Prod

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
�
RandomStandardNormal

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
~
RandomUniform

shape"T
output"dtype"
seedint "
seed2int "
dtypetype:
2"
Ttype:
2	�
a
Range
start"Tidx
limit"Tidx
delta"Tidx
output"Tidx"
Tidxtype0:	
2	
>
RealDiv
x"T
y"T
z"T"
Ttype:
2	
5

Reciprocal
x"T
y"T"
Ttype:

2	
E
Relu
features"T
activations"T"
Ttype:
2	
V
ReluGrad
	gradients"T
features"T
	backprops"T"
Ttype:
2	
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
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
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
9
Softmax
logits"T
softmax"T"
Ttype:
2
2
StopGradient

input"T
output"T"	
Ttype
�
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
:
Sub
x"T
y"T
z"T"
Ttype:
2	
�
Sum

input"T
reduction_indices"Tidx
output"T"
	keep_dimsbool( " 
Ttype:
2	"
Tidxtype0:
2	
c
Tile

input"T
	multiples"
Tmultiples
output"T"	
Ttype"

Tmultiplestype0:
2	
P
	Transpose
x"T
perm"Tperm
y"T"	
Ttype"
Tpermtype0:
2	
s

VariableV2
ref"dtype�"
shapeshape"
dtypetype"
	containerstring "
shared_namestring �
&
	ZerosLike
x"T
y"T"	
Ttype"serve*1.15.32v1.15.2-30-g4386a66��
n
PlaceholderPlaceholder*
dtype0*'
_output_shapes
:���������*
shape:���������
p
Placeholder_1Placeholder*
shape:���������*
dtype0*'
_output_shapes
:���������
p
Placeholder_2Placeholder*
dtype0*'
_output_shapes
:���������*
shape:���������
h
Placeholder_3Placeholder*
dtype0*#
_output_shapes
:���������*
shape:���������
h
Placeholder_4Placeholder*
shape:���������*
dtype0*#
_output_shapes
:���������
H
Cast/xConst*
value	B :*
dtype0*
_output_shapes
: 
T
CastCastCast/x*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
1
LogLogCast*
T0*
_output_shapes
: 
N
Placeholder_5Placeholder*
dtype0*
_output_shapes
: *
shape: 
?
mulMulLogPlaceholder_5*
T0*
_output_shapes
: 
J
ConstConst*
valueB
 *    *
dtype0*
_output_shapes
: 
\
log_alpha/initial_valueConst*
valueB
 *    *
dtype0*
_output_shapes
: 
m
	log_alpha
VariableV2*
_output_shapes
: *
	container *
shape: *
shared_name *
dtype0
�
log_alpha/AssignAssign	log_alphalog_alpha/initial_value*
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: 
d
log_alpha/readIdentity	log_alpha*
_class
loc:@log_alpha*
_output_shapes
: *
T0
;
ExpExplog_alpha/read*
_output_shapes
: *
T0
�
5main/pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *'
_class
loc:@main/pi/dense/kernel*
dtype0*
_output_shapes
:
�
3main/pi/dense/kernel/Initializer/random_uniform/minConst*
dtype0*
_output_shapes
: *
valueB
 *0��*'
_class
loc:@main/pi/dense/kernel
�
3main/pi/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *0�>*'
_class
loc:@main/pi/dense/kernel*
dtype0
�
=main/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5main/pi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*'
_class
loc:@main/pi/dense/kernel*
seed2
�
3main/pi/dense/kernel/Initializer/random_uniform/subSub3main/pi/dense/kernel/Initializer/random_uniform/max3main/pi/dense/kernel/Initializer/random_uniform/min*'
_class
loc:@main/pi/dense/kernel*
_output_shapes
: *
T0
�
3main/pi/dense/kernel/Initializer/random_uniform/mulMul=main/pi/dense/kernel/Initializer/random_uniform/RandomUniform3main/pi/dense/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*'
_class
loc:@main/pi/dense/kernel
�
/main/pi/dense/kernel/Initializer/random_uniformAdd3main/pi/dense/kernel/Initializer/random_uniform/mul3main/pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes

:@*
T0*'
_class
loc:@main/pi/dense/kernel
�
main/pi/dense/kernel
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *'
_class
loc:@main/pi/dense/kernel*
	container *
shape
:@
�
main/pi/dense/kernel/AssignAssignmain/pi/dense/kernel/main/pi/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
main/pi/dense/kernel/readIdentitymain/pi/dense/kernel*'
_class
loc:@main/pi/dense/kernel*
_output_shapes

:@*
T0
�
$main/pi/dense/bias/Initializer/zerosConst*
valueB@*    *%
_class
loc:@main/pi/dense/bias*
dtype0*
_output_shapes
:@
�
main/pi/dense/bias
VariableV2*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
main/pi/dense/bias/AssignAssignmain/pi/dense/bias$main/pi/dense/bias/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(
�
main/pi/dense/bias/readIdentitymain/pi/dense/bias*
_output_shapes
:@*
T0*%
_class
loc:@main/pi/dense/bias
�
main/pi/dense/MatMulMatMulPlaceholdermain/pi/dense/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
main/pi/dense/BiasAddBiasAddmain/pi/dense/MatMulmain/pi/dense/bias/read*
data_formatNHWC*'
_output_shapes
:���������@*
T0
c
main/pi/dense/ReluRelumain/pi/dense/BiasAdd*'
_output_shapes
:���������@*
T0
�
7main/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
:
�
5main/pi/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *׳]�*)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
�
5main/pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*)
_class
loc:@main/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
�
?main/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*)
_class
loc:@main/pi/dense_1/kernel*
seed2$
�
5main/pi/dense_1/kernel/Initializer/random_uniform/subSub5main/pi/dense_1/kernel/Initializer/random_uniform/max5main/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes
: 
�
5main/pi/dense_1/kernel/Initializer/random_uniform/mulMul?main/pi/dense_1/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_1/kernel/Initializer/random_uniform/sub*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
1main/pi/dense_1/kernel/Initializer/random_uniformAdd5main/pi/dense_1/kernel/Initializer/random_uniform/mul5main/pi/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes

:@@
�
main/pi/dense_1/kernel
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *)
_class
loc:@main/pi/dense_1/kernel*
	container 
�
main/pi/dense_1/kernel/AssignAssignmain/pi/dense_1/kernel1main/pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
main/pi/dense_1/kernel/readIdentitymain/pi/dense_1/kernel*
_output_shapes

:@@*
T0*)
_class
loc:@main/pi/dense_1/kernel
�
&main/pi/dense_1/bias/Initializer/zerosConst*
valueB@*    *'
_class
loc:@main/pi/dense_1/bias*
dtype0*
_output_shapes
:@
�
main/pi/dense_1/bias
VariableV2*
_output_shapes
:@*
shared_name *'
_class
loc:@main/pi/dense_1/bias*
	container *
shape:@*
dtype0
�
main/pi/dense_1/bias/AssignAssignmain/pi/dense_1/bias&main/pi/dense_1/bias/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(
�
main/pi/dense_1/bias/readIdentitymain/pi/dense_1/bias*
_output_shapes
:@*
T0*'
_class
loc:@main/pi/dense_1/bias
�
main/pi/dense_1/MatMulMatMulmain/pi/dense/Relumain/pi/dense_1/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
main/pi/dense_1/BiasAddBiasAddmain/pi/dense_1/MatMulmain/pi/dense_1/bias/read*'
_output_shapes
:���������@*
T0*
data_formatNHWC
g
main/pi/dense_1/ReluRelumain/pi/dense_1/BiasAdd*'
_output_shapes
:���������@*
T0
�
7main/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *)
_class
loc:@main/pi/dense_2/kernel*
dtype0*
_output_shapes
:
�
5main/pi/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *�_��*)
_class
loc:@main/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
�
5main/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *�_�>*)
_class
loc:@main/pi/dense_2/kernel*
dtype0*
_output_shapes
: 
�
?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/pi/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*)
_class
loc:@main/pi/dense_2/kernel*
seed25
�
5main/pi/dense_2/kernel/Initializer/random_uniform/subSub5main/pi/dense_2/kernel/Initializer/random_uniform/max5main/pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@main/pi/dense_2/kernel
�
5main/pi/dense_2/kernel/Initializer/random_uniform/mulMul?main/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/pi/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*)
_class
loc:@main/pi/dense_2/kernel
�
1main/pi/dense_2/kernel/Initializer/random_uniformAdd5main/pi/dense_2/kernel/Initializer/random_uniform/mul5main/pi/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes

:@*
T0*)
_class
loc:@main/pi/dense_2/kernel
�
main/pi/dense_2/kernel
VariableV2*)
_class
loc:@main/pi/dense_2/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
main/pi/dense_2/kernel/AssignAssignmain/pi/dense_2/kernel1main/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
main/pi/dense_2/kernel/readIdentitymain/pi/dense_2/kernel*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes

:@*
T0
�
&main/pi/dense_2/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@main/pi/dense_2/bias*
dtype0*
_output_shapes
:
�
main/pi/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/pi/dense_2/bias*
	container *
shape:
�
main/pi/dense_2/bias/AssignAssignmain/pi/dense_2/bias&main/pi/dense_2/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
main/pi/dense_2/bias/readIdentitymain/pi/dense_2/bias*
T0*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:
�
main/pi/dense_2/MatMulMatMulmain/pi/dense_1/Relumain/pi/dense_2/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
main/pi/dense_2/BiasAddBiasAddmain/pi/dense_2/MatMulmain/pi/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
e
main/pi/SoftmaxSoftmaxmain/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:���������
k
main/pi/LogSoftmax
LogSoftmaxmain/pi/dense_2/BiasAdd*'
_output_shapes
:���������*
T0
c
main/pi/ArgMax/dimensionConst*
_output_shapes
: *
valueB :
���������*
dtype0
�
main/pi/ArgMaxArgMaxmain/pi/dense_2/BiasAddmain/pi/ArgMax/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
o
main/pi/Categorical/probsSoftmaxmain/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:���������
`
main/pi/Categorical/batch_rankConst*
value	B :*
dtype0*
_output_shapes
: 
w
 main/pi/Categorical/logits_shapeShapemain/pi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
`
main/pi/Categorical/event_sizeConst*
value	B :*
dtype0*
_output_shapes
: 
}
3main/pi/Categorical/batch_shape/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB: 
�
5main/pi/Categorical/batch_shape/strided_slice/stack_1Const*
valueB:
���������*
dtype0*
_output_shapes
:

5main/pi/Categorical/batch_shape/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
-main/pi/Categorical/batch_shape/strided_sliceStridedSlice main/pi/Categorical/logits_shape3main/pi/Categorical/batch_shape/strided_slice/stack5main/pi/Categorical/batch_shape/strided_slice/stack_15main/pi/Categorical/batch_shape/strided_slice/stack_2*
ellipsis_mask *

begin_mask*
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask 
j
'main/pi/Categorical/sample/sample_shapeConst*
dtype0*
_output_shapes
: *
valueB 
�
>main/pi/Categorical/sample/multinomial/Multinomial/num_samplesConst*
_output_shapes
: *
value	B :*
dtype0
�
2main/pi/Categorical/sample/multinomial/MultinomialMultinomialmain/pi/dense_2/BiasAdd>main/pi/Categorical/sample/multinomial/Multinomial/num_samples*'
_output_shapes
:���������*
seed2O*

seed*
output_dtype0*
T0
z
)main/pi/Categorical/sample/transpose/permConst*
dtype0*
_output_shapes
:*
valueB"       
�
$main/pi/Categorical/sample/transpose	Transpose2main/pi/Categorical/sample/multinomial/Multinomial)main/pi/Categorical/sample/transpose/perm*
T0*'
_output_shapes
:���������*
Tperm0
�
/main/pi/Categorical/batch_shape_tensor/IdentityIdentity-main/pi/Categorical/batch_shape/strided_slice*
T0*
_output_shapes
:
t
*main/pi/Categorical/sample/concat/values_0Const*
_output_shapes
:*
valueB:*
dtype0
h
&main/pi/Categorical/sample/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
!main/pi/Categorical/sample/concatConcatV2*main/pi/Categorical/sample/concat/values_0/main/pi/Categorical/batch_shape_tensor/Identity&main/pi/Categorical/sample/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
"main/pi/Categorical/sample/ReshapeReshape$main/pi/Categorical/sample/transpose!main/pi/Categorical/sample/concat*
T0*
Tshape0*'
_output_shapes
:���������
�
 main/pi/Categorical/sample/ShapeShape"main/pi/Categorical/sample/Reshape*
T0*
out_type0*
_output_shapes
:
x
.main/pi/Categorical/sample/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
z
0main/pi/Categorical/sample/strided_slice/stack_1Const*
dtype0*
_output_shapes
:*
valueB: 
z
0main/pi/Categorical/sample/strided_slice/stack_2Const*
dtype0*
_output_shapes
:*
valueB:
�
(main/pi/Categorical/sample/strided_sliceStridedSlice main/pi/Categorical/sample/Shape.main/pi/Categorical/sample/strided_slice/stack0main/pi/Categorical/sample/strided_slice/stack_10main/pi/Categorical/sample/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
j
(main/pi/Categorical/sample/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#main/pi/Categorical/sample/concat_1ConcatV2'main/pi/Categorical/sample/sample_shape(main/pi/Categorical/sample/strided_slice(main/pi/Categorical/sample/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
$main/pi/Categorical/sample/Reshape_1Reshape"main/pi/Categorical/sample/Reshape#main/pi/Categorical/sample/concat_1*#
_output_shapes
:���������*
T0*
Tshape0
�
5main/q1/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *'
_class
loc:@main/q1/dense/kernel*
dtype0*
_output_shapes
:
�
3main/q1/dense/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *0��*'
_class
loc:@main/q1/dense/kernel*
dtype0
�
3main/q1/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *0�>*'
_class
loc:@main/q1/dense/kernel*
dtype0
�
=main/q1/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5main/q1/dense/kernel/Initializer/random_uniform/shape*'
_class
loc:@main/q1/dense/kernel*
seed2c*
dtype0*
_output_shapes

:@*

seed*
T0
�
3main/q1/dense/kernel/Initializer/random_uniform/subSub3main/q1/dense/kernel/Initializer/random_uniform/max3main/q1/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes
: 
�
3main/q1/dense/kernel/Initializer/random_uniform/mulMul=main/q1/dense/kernel/Initializer/random_uniform/RandomUniform3main/q1/dense/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*'
_class
loc:@main/q1/dense/kernel
�
/main/q1/dense/kernel/Initializer/random_uniformAdd3main/q1/dense/kernel/Initializer/random_uniform/mul3main/q1/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes

:@
�
main/q1/dense/kernel
VariableV2*
shared_name *'
_class
loc:@main/q1/dense/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
main/q1/dense/kernel/AssignAssignmain/q1/dense/kernel/main/q1/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
main/q1/dense/kernel/readIdentitymain/q1/dense/kernel*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes

:@
�
$main/q1/dense/bias/Initializer/zerosConst*
dtype0*
_output_shapes
:@*
valueB@*    *%
_class
loc:@main/q1/dense/bias
�
main/q1/dense/bias
VariableV2*
shared_name *%
_class
loc:@main/q1/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
main/q1/dense/bias/AssignAssignmain/q1/dense/bias$main/q1/dense/bias/Initializer/zeros*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@
�
main/q1/dense/bias/readIdentitymain/q1/dense/bias*
_output_shapes
:@*
T0*%
_class
loc:@main/q1/dense/bias
�
main/q1/dense/MatMulMatMulPlaceholdermain/q1/dense/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
main/q1/dense/BiasAddBiasAddmain/q1/dense/MatMulmain/q1/dense/bias/read*
data_formatNHWC*'
_output_shapes
:���������@*
T0
c
main/q1/dense/ReluRelumain/q1/dense/BiasAdd*'
_output_shapes
:���������@*
T0
�
7main/q1/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *)
_class
loc:@main/q1/dense_1/kernel*
dtype0*
_output_shapes
:
�
5main/q1/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *׳]�*)
_class
loc:@main/q1/dense_1/kernel*
dtype0*
_output_shapes
: 
�
5main/q1/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*)
_class
loc:@main/q1/dense_1/kernel*
dtype0*
_output_shapes
: 
�
?main/q1/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q1/dense_1/kernel/Initializer/random_uniform/shape*)
_class
loc:@main/q1/dense_1/kernel*
seed2t*
dtype0*
_output_shapes

:@@*

seed*
T0
�
5main/q1/dense_1/kernel/Initializer/random_uniform/subSub5main/q1/dense_1/kernel/Initializer/random_uniform/max5main/q1/dense_1/kernel/Initializer/random_uniform/min*)
_class
loc:@main/q1/dense_1/kernel*
_output_shapes
: *
T0
�
5main/q1/dense_1/kernel/Initializer/random_uniform/mulMul?main/q1/dense_1/kernel/Initializer/random_uniform/RandomUniform5main/q1/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/q1/dense_1/kernel*
_output_shapes

:@@
�
1main/q1/dense_1/kernel/Initializer/random_uniformAdd5main/q1/dense_1/kernel/Initializer/random_uniform/mul5main/q1/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*)
_class
loc:@main/q1/dense_1/kernel
�
main/q1/dense_1/kernel
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *)
_class
loc:@main/q1/dense_1/kernel*
	container 
�
main/q1/dense_1/kernel/AssignAssignmain/q1/dense_1/kernel1main/q1/dense_1/kernel/Initializer/random_uniform*
_output_shapes

:@@*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(
�
main/q1/dense_1/kernel/readIdentitymain/q1/dense_1/kernel*)
_class
loc:@main/q1/dense_1/kernel*
_output_shapes

:@@*
T0
�
&main/q1/dense_1/bias/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *'
_class
loc:@main/q1/dense_1/bias*
dtype0
�
main/q1/dense_1/bias
VariableV2*
shared_name *'
_class
loc:@main/q1/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
main/q1/dense_1/bias/AssignAssignmain/q1/dense_1/bias&main/q1/dense_1/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
main/q1/dense_1/bias/readIdentitymain/q1/dense_1/bias*
_output_shapes
:@*
T0*'
_class
loc:@main/q1/dense_1/bias
�
main/q1/dense_1/MatMulMatMulmain/q1/dense/Relumain/q1/dense_1/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
main/q1/dense_1/BiasAddBiasAddmain/q1/dense_1/MatMulmain/q1/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������@
g
main/q1/dense_1/ReluRelumain/q1/dense_1/BiasAdd*
T0*'
_output_shapes
:���������@
�
7main/q1/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *)
_class
loc:@main/q1/dense_2/kernel*
dtype0*
_output_shapes
:
�
5main/q1/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *�_��*)
_class
loc:@main/q1/dense_2/kernel*
dtype0*
_output_shapes
: 
�
5main/q1/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *�_�>*)
_class
loc:@main/q1/dense_2/kernel*
dtype0*
_output_shapes
: 
�
?main/q1/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q1/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*)
_class
loc:@main/q1/dense_2/kernel*
seed2�
�
5main/q1/dense_2/kernel/Initializer/random_uniform/subSub5main/q1/dense_2/kernel/Initializer/random_uniform/max5main/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes
: 
�
5main/q1/dense_2/kernel/Initializer/random_uniform/mulMul?main/q1/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/q1/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*)
_class
loc:@main/q1/dense_2/kernel
�
1main/q1/dense_2/kernel/Initializer/random_uniformAdd5main/q1/dense_2/kernel/Initializer/random_uniform/mul5main/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes

:@
�
main/q1/dense_2/kernel
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *)
_class
loc:@main/q1/dense_2/kernel*
	container 
�
main/q1/dense_2/kernel/AssignAssignmain/q1/dense_2/kernel1main/q1/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel
�
main/q1/dense_2/kernel/readIdentitymain/q1/dense_2/kernel*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes

:@*
T0
�
&main/q1/dense_2/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@main/q1/dense_2/bias*
dtype0*
_output_shapes
:
�
main/q1/dense_2/bias
VariableV2*
shared_name *'
_class
loc:@main/q1/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
main/q1/dense_2/bias/AssignAssignmain/q1/dense_2/bias&main/q1/dense_2/bias/Initializer/zeros*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
main/q1/dense_2/bias/readIdentitymain/q1/dense_2/bias*
_output_shapes
:*
T0*'
_class
loc:@main/q1/dense_2/bias
�
main/q1/dense_2/MatMulMatMulmain/q1/dense_1/Relumain/q1/dense_2/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
main/q1/dense_2/BiasAddBiasAddmain/q1/dense_2/MatMulmain/q1/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
�
5main/q2/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *'
_class
loc:@main/q2/dense/kernel*
dtype0*
_output_shapes
:
�
3main/q2/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *0��*'
_class
loc:@main/q2/dense/kernel*
dtype0*
_output_shapes
: 
�
3main/q2/dense/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *0�>*'
_class
loc:@main/q2/dense/kernel*
dtype0
�
=main/q2/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform5main/q2/dense/kernel/Initializer/random_uniform/shape*
_output_shapes

:@*

seed*
T0*'
_class
loc:@main/q2/dense/kernel*
seed2�*
dtype0
�
3main/q2/dense/kernel/Initializer/random_uniform/subSub3main/q2/dense/kernel/Initializer/random_uniform/max3main/q2/dense/kernel/Initializer/random_uniform/min*
T0*'
_class
loc:@main/q2/dense/kernel*
_output_shapes
: 
�
3main/q2/dense/kernel/Initializer/random_uniform/mulMul=main/q2/dense/kernel/Initializer/random_uniform/RandomUniform3main/q2/dense/kernel/Initializer/random_uniform/sub*
T0*'
_class
loc:@main/q2/dense/kernel*
_output_shapes

:@
�
/main/q2/dense/kernel/Initializer/random_uniformAdd3main/q2/dense/kernel/Initializer/random_uniform/mul3main/q2/dense/kernel/Initializer/random_uniform/min*
_output_shapes

:@*
T0*'
_class
loc:@main/q2/dense/kernel
�
main/q2/dense/kernel
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *'
_class
loc:@main/q2/dense/kernel*
	container *
shape
:@
�
main/q2/dense/kernel/AssignAssignmain/q2/dense/kernel/main/q2/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
main/q2/dense/kernel/readIdentitymain/q2/dense/kernel*'
_class
loc:@main/q2/dense/kernel*
_output_shapes

:@*
T0
�
$main/q2/dense/bias/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *%
_class
loc:@main/q2/dense/bias*
dtype0
�
main/q2/dense/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *%
_class
loc:@main/q2/dense/bias*
	container *
shape:@
�
main/q2/dense/bias/AssignAssignmain/q2/dense/bias$main/q2/dense/bias/Initializer/zeros*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
main/q2/dense/bias/readIdentitymain/q2/dense/bias*
T0*%
_class
loc:@main/q2/dense/bias*
_output_shapes
:@
�
main/q2/dense/MatMulMatMulPlaceholdermain/q2/dense/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
main/q2/dense/BiasAddBiasAddmain/q2/dense/MatMulmain/q2/dense/bias/read*'
_output_shapes
:���������@*
T0*
data_formatNHWC
c
main/q2/dense/ReluRelumain/q2/dense/BiasAdd*
T0*'
_output_shapes
:���������@
�
7main/q2/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"@   @   *)
_class
loc:@main/q2/dense_1/kernel*
dtype0
�
5main/q2/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *׳]�*)
_class
loc:@main/q2/dense_1/kernel*
dtype0*
_output_shapes
: 
�
5main/q2/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*)
_class
loc:@main/q2/dense_1/kernel*
dtype0*
_output_shapes
: 
�
?main/q2/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q2/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*)
_class
loc:@main/q2/dense_1/kernel*
seed2�
�
5main/q2/dense_1/kernel/Initializer/random_uniform/subSub5main/q2/dense_1/kernel/Initializer/random_uniform/max5main/q2/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q2/dense_1/kernel*
_output_shapes
: 
�
5main/q2/dense_1/kernel/Initializer/random_uniform/mulMul?main/q2/dense_1/kernel/Initializer/random_uniform/RandomUniform5main/q2/dense_1/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/q2/dense_1/kernel*
_output_shapes

:@@
�
1main/q2/dense_1/kernel/Initializer/random_uniformAdd5main/q2/dense_1/kernel/Initializer/random_uniform/mul5main/q2/dense_1/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q2/dense_1/kernel*
_output_shapes

:@@
�
main/q2/dense_1/kernel
VariableV2*
shared_name *)
_class
loc:@main/q2/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
main/q2/dense_1/kernel/AssignAssignmain/q2/dense_1/kernel1main/q2/dense_1/kernel/Initializer/random_uniform*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
main/q2/dense_1/kernel/readIdentitymain/q2/dense_1/kernel*)
_class
loc:@main/q2/dense_1/kernel*
_output_shapes

:@@*
T0
�
&main/q2/dense_1/bias/Initializer/zerosConst*
valueB@*    *'
_class
loc:@main/q2/dense_1/bias*
dtype0*
_output_shapes
:@
�
main/q2/dense_1/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@main/q2/dense_1/bias*
	container *
shape:@
�
main/q2/dense_1/bias/AssignAssignmain/q2/dense_1/bias&main/q2/dense_1/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
main/q2/dense_1/bias/readIdentitymain/q2/dense_1/bias*
_output_shapes
:@*
T0*'
_class
loc:@main/q2/dense_1/bias
�
main/q2/dense_1/MatMulMatMulmain/q2/dense/Relumain/q2/dense_1/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
main/q2/dense_1/BiasAddBiasAddmain/q2/dense_1/MatMulmain/q2/dense_1/bias/read*'
_output_shapes
:���������@*
T0*
data_formatNHWC
g
main/q2/dense_1/ReluRelumain/q2/dense_1/BiasAdd*
T0*'
_output_shapes
:���������@
�
7main/q2/dense_2/kernel/Initializer/random_uniform/shapeConst*
dtype0*
_output_shapes
:*
valueB"@      *)
_class
loc:@main/q2/dense_2/kernel
�
5main/q2/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *�_��*)
_class
loc:@main/q2/dense_2/kernel*
dtype0*
_output_shapes
: 
�
5main/q2/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *�_�>*)
_class
loc:@main/q2/dense_2/kernel*
dtype0*
_output_shapes
: 
�
?main/q2/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform7main/q2/dense_2/kernel/Initializer/random_uniform/shape*
T0*)
_class
loc:@main/q2/dense_2/kernel*
seed2�*
dtype0*
_output_shapes

:@*

seed
�
5main/q2/dense_2/kernel/Initializer/random_uniform/subSub5main/q2/dense_2/kernel/Initializer/random_uniform/max5main/q2/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@main/q2/dense_2/kernel
�
5main/q2/dense_2/kernel/Initializer/random_uniform/mulMul?main/q2/dense_2/kernel/Initializer/random_uniform/RandomUniform5main/q2/dense_2/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes

:@
�
1main/q2/dense_2/kernel/Initializer/random_uniformAdd5main/q2/dense_2/kernel/Initializer/random_uniform/mul5main/q2/dense_2/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes

:@
�
main/q2/dense_2/kernel
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *)
_class
loc:@main/q2/dense_2/kernel
�
main/q2/dense_2/kernel/AssignAssignmain/q2/dense_2/kernel1main/q2/dense_2/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel
�
main/q2/dense_2/kernel/readIdentitymain/q2/dense_2/kernel*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes

:@*
T0
�
&main/q2/dense_2/bias/Initializer/zerosConst*
valueB*    *'
_class
loc:@main/q2/dense_2/bias*
dtype0*
_output_shapes
:
�
main/q2/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/q2/dense_2/bias*
	container *
shape:
�
main/q2/dense_2/bias/AssignAssignmain/q2/dense_2/bias&main/q2/dense_2/bias/Initializer/zeros*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
main/q2/dense_2/bias/readIdentitymain/q2/dense_2/bias*
_output_shapes
:*
T0*'
_class
loc:@main/q2/dense_2/bias
�
main/q2/dense_2/MatMulMatMulmain/q2/dense_1/Relumain/q2/dense_2/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
main/q2/dense_2/BiasAddBiasAddmain/q2/dense_2/MatMulmain/q2/dense_2/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
�
main_1/pi/dense/MatMulMatMulPlaceholder_2main/pi/dense/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
main_1/pi/dense/BiasAddBiasAddmain_1/pi/dense/MatMulmain/pi/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������@
g
main_1/pi/dense/ReluRelumain_1/pi/dense/BiasAdd*'
_output_shapes
:���������@*
T0
�
main_1/pi/dense_1/MatMulMatMulmain_1/pi/dense/Relumain/pi/dense_1/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
main_1/pi/dense_1/BiasAddBiasAddmain_1/pi/dense_1/MatMulmain/pi/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������@
k
main_1/pi/dense_1/ReluRelumain_1/pi/dense_1/BiasAdd*
T0*'
_output_shapes
:���������@
�
main_1/pi/dense_2/MatMulMatMulmain_1/pi/dense_1/Relumain/pi/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
main_1/pi/dense_2/BiasAddBiasAddmain_1/pi/dense_2/MatMulmain/pi/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
i
main_1/pi/SoftmaxSoftmaxmain_1/pi/dense_2/BiasAdd*'
_output_shapes
:���������*
T0
o
main_1/pi/LogSoftmax
LogSoftmaxmain_1/pi/dense_2/BiasAdd*'
_output_shapes
:���������*
T0
e
main_1/pi/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
main_1/pi/ArgMaxArgMaxmain_1/pi/dense_2/BiasAddmain_1/pi/ArgMax/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
s
main_1/pi/Categorical/probsSoftmaxmain_1/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:���������
b
 main_1/pi/Categorical/batch_rankConst*
value	B :*
dtype0*
_output_shapes
: 
{
"main_1/pi/Categorical/logits_shapeShapemain_1/pi/dense_2/BiasAdd*
T0*
out_type0*
_output_shapes
:
b
 main_1/pi/Categorical/event_sizeConst*
value	B :*
dtype0*
_output_shapes
: 

5main_1/pi/Categorical/batch_shape/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
7main_1/pi/Categorical/batch_shape/strided_slice/stack_1Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
7main_1/pi/Categorical/batch_shape/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
/main_1/pi/Categorical/batch_shape/strided_sliceStridedSlice"main_1/pi/Categorical/logits_shape5main_1/pi/Categorical/batch_shape/strided_slice/stack7main_1/pi/Categorical/batch_shape/strided_slice/stack_17main_1/pi/Categorical/batch_shape/strided_slice/stack_2*
new_axis_mask *
end_mask *
_output_shapes
:*
Index0*
T0*
shrink_axis_mask *
ellipsis_mask *

begin_mask
l
)main_1/pi/Categorical/sample/sample_shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
@main_1/pi/Categorical/sample/multinomial/Multinomial/num_samplesConst*
value	B :*
dtype0*
_output_shapes
: 
�
4main_1/pi/Categorical/sample/multinomial/MultinomialMultinomialmain_1/pi/dense_2/BiasAdd@main_1/pi/Categorical/sample/multinomial/Multinomial/num_samples*
T0*'
_output_shapes
:���������*
seed2�*

seed*
output_dtype0
|
+main_1/pi/Categorical/sample/transpose/permConst*
valueB"       *
dtype0*
_output_shapes
:
�
&main_1/pi/Categorical/sample/transpose	Transpose4main_1/pi/Categorical/sample/multinomial/Multinomial+main_1/pi/Categorical/sample/transpose/perm*'
_output_shapes
:���������*
Tperm0*
T0
�
1main_1/pi/Categorical/batch_shape_tensor/IdentityIdentity/main_1/pi/Categorical/batch_shape/strided_slice*
T0*
_output_shapes
:
v
,main_1/pi/Categorical/sample/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
j
(main_1/pi/Categorical/sample/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#main_1/pi/Categorical/sample/concatConcatV2,main_1/pi/Categorical/sample/concat/values_01main_1/pi/Categorical/batch_shape_tensor/Identity(main_1/pi/Categorical/sample/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
$main_1/pi/Categorical/sample/ReshapeReshape&main_1/pi/Categorical/sample/transpose#main_1/pi/Categorical/sample/concat*
T0*
Tshape0*'
_output_shapes
:���������
�
"main_1/pi/Categorical/sample/ShapeShape$main_1/pi/Categorical/sample/Reshape*
_output_shapes
:*
T0*
out_type0
z
0main_1/pi/Categorical/sample/strided_slice/stackConst*
_output_shapes
:*
valueB:*
dtype0
|
2main_1/pi/Categorical/sample/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
|
2main_1/pi/Categorical/sample/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
*main_1/pi/Categorical/sample/strided_sliceStridedSlice"main_1/pi/Categorical/sample/Shape0main_1/pi/Categorical/sample/strided_slice/stack2main_1/pi/Categorical/sample/strided_slice/stack_12main_1/pi/Categorical/sample/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
l
*main_1/pi/Categorical/sample/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
%main_1/pi/Categorical/sample/concat_1ConcatV2)main_1/pi/Categorical/sample/sample_shape*main_1/pi/Categorical/sample/strided_slice*main_1/pi/Categorical/sample/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
&main_1/pi/Categorical/sample/Reshape_1Reshape$main_1/pi/Categorical/sample/Reshape%main_1/pi/Categorical/sample/concat_1*
T0*
Tshape0*#
_output_shapes
:���������
�
main_1/q1/dense/MatMulMatMulPlaceholder_2main/q1/dense/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
main_1/q1/dense/BiasAddBiasAddmain_1/q1/dense/MatMulmain/q1/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������@
g
main_1/q1/dense/ReluRelumain_1/q1/dense/BiasAdd*
T0*'
_output_shapes
:���������@
�
main_1/q1/dense_1/MatMulMatMulmain_1/q1/dense/Relumain/q1/dense_1/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
main_1/q1/dense_1/BiasAddBiasAddmain_1/q1/dense_1/MatMulmain/q1/dense_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������@*
T0
k
main_1/q1/dense_1/ReluRelumain_1/q1/dense_1/BiasAdd*
T0*'
_output_shapes
:���������@
�
main_1/q1/dense_2/MatMulMatMulmain_1/q1/dense_1/Relumain/q1/dense_2/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
main_1/q1/dense_2/BiasAddBiasAddmain_1/q1/dense_2/MatMulmain/q1/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
�
main_1/q2/dense/MatMulMatMulPlaceholder_2main/q2/dense/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
main_1/q2/dense/BiasAddBiasAddmain_1/q2/dense/MatMulmain/q2/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������@
g
main_1/q2/dense/ReluRelumain_1/q2/dense/BiasAdd*'
_output_shapes
:���������@*
T0
�
main_1/q2/dense_1/MatMulMatMulmain_1/q2/dense/Relumain/q2/dense_1/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
main_1/q2/dense_1/BiasAddBiasAddmain_1/q2/dense_1/MatMulmain/q2/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������@
k
main_1/q2/dense_1/ReluRelumain_1/q2/dense_1/BiasAdd*'
_output_shapes
:���������@*
T0
�
main_1/q2/dense_2/MatMulMatMulmain_1/q2/dense_1/Relumain/q2/dense_2/kernel/read*
transpose_b( *
T0*'
_output_shapes
:���������*
transpose_a( 
�
main_1/q2/dense_2/BiasAddBiasAddmain_1/q2/dense_2/MatMulmain/q2/dense_2/bias/read*'
_output_shapes
:���������*
T0*
data_formatNHWC
�
7target/pi/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *)
_class
loc:@target/pi/dense/kernel*
dtype0*
_output_shapes
:
�
5target/pi/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *0��*)
_class
loc:@target/pi/dense/kernel*
dtype0*
_output_shapes
: 
�
5target/pi/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *0�>*)
_class
loc:@target/pi/dense/kernel*
dtype0*
_output_shapes
: 
�
?target/pi/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7target/pi/dense/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*)
_class
loc:@target/pi/dense/kernel*
seed2�
�
5target/pi/dense/kernel/Initializer/random_uniform/subSub5target/pi/dense/kernel/Initializer/random_uniform/max5target/pi/dense/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*)
_class
loc:@target/pi/dense/kernel
�
5target/pi/dense/kernel/Initializer/random_uniform/mulMul?target/pi/dense/kernel/Initializer/random_uniform/RandomUniform5target/pi/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes

:@
�
1target/pi/dense/kernel/Initializer/random_uniformAdd5target/pi/dense/kernel/Initializer/random_uniform/mul5target/pi/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@target/pi/dense/kernel*
_output_shapes

:@
�
target/pi/dense/kernel
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *)
_class
loc:@target/pi/dense/kernel
�
target/pi/dense/kernel/AssignAssigntarget/pi/dense/kernel1target/pi/dense/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel
�
target/pi/dense/kernel/readIdentitytarget/pi/dense/kernel*)
_class
loc:@target/pi/dense/kernel*
_output_shapes

:@*
T0
�
&target/pi/dense/bias/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *'
_class
loc:@target/pi/dense/bias*
dtype0
�
target/pi/dense/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@target/pi/dense/bias*
	container *
shape:@
�
target/pi/dense/bias/AssignAssigntarget/pi/dense/bias&target/pi/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
target/pi/dense/bias/readIdentitytarget/pi/dense/bias*'
_class
loc:@target/pi/dense/bias*
_output_shapes
:@*
T0
�
target/pi/dense/MatMulMatMulPlaceholder_2target/pi/dense/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
target/pi/dense/BiasAddBiasAddtarget/pi/dense/MatMultarget/pi/dense/bias/read*
data_formatNHWC*'
_output_shapes
:���������@*
T0
g
target/pi/dense/ReluRelutarget/pi/dense/BiasAdd*
T0*'
_output_shapes
:���������@
�
9target/pi/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
_output_shapes
:
�
7target/pi/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *׳]�*+
_class!
loc:@target/pi/dense_1/kernel*
dtype0
�
7target/pi/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*+
_class!
loc:@target/pi/dense_1/kernel*
dtype0*
_output_shapes
: 
�
Atarget/pi/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_1/kernel/Initializer/random_uniform/shape*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
seed2�*
dtype0*
_output_shapes

:@@*

seed
�
7target/pi/dense_1/kernel/Initializer/random_uniform/subSub7target/pi/dense_1/kernel/Initializer/random_uniform/max7target/pi/dense_1/kernel/Initializer/random_uniform/min*+
_class!
loc:@target/pi/dense_1/kernel*
_output_shapes
: *
T0
�
7target/pi/dense_1/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_1/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
_output_shapes

:@@
�
3target/pi/dense_1/kernel/Initializer/random_uniformAdd7target/pi/dense_1/kernel/Initializer/random_uniform/mul7target/pi/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*+
_class!
loc:@target/pi/dense_1/kernel
�
target/pi/dense_1/kernel
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *+
_class!
loc:@target/pi/dense_1/kernel*
	container 
�
target/pi/dense_1/kernel/AssignAssigntarget/pi/dense_1/kernel3target/pi/dense_1/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
target/pi/dense_1/kernel/readIdentitytarget/pi/dense_1/kernel*
_output_shapes

:@@*
T0*+
_class!
loc:@target/pi/dense_1/kernel
�
(target/pi/dense_1/bias/Initializer/zerosConst*
valueB@*    *)
_class
loc:@target/pi/dense_1/bias*
dtype0*
_output_shapes
:@
�
target/pi/dense_1/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *)
_class
loc:@target/pi/dense_1/bias*
	container *
shape:@
�
target/pi/dense_1/bias/AssignAssigntarget/pi/dense_1/bias(target/pi/dense_1/bias/Initializer/zeros*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
target/pi/dense_1/bias/readIdentitytarget/pi/dense_1/bias*
T0*)
_class
loc:@target/pi/dense_1/bias*
_output_shapes
:@
�
target/pi/dense_1/MatMulMatMultarget/pi/dense/Relutarget/pi/dense_1/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
target/pi/dense_1/BiasAddBiasAddtarget/pi/dense_1/MatMultarget/pi/dense_1/bias/read*'
_output_shapes
:���������@*
T0*
data_formatNHWC
k
target/pi/dense_1/ReluRelutarget/pi/dense_1/BiasAdd*
T0*'
_output_shapes
:���������@
�
9target/pi/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *+
_class!
loc:@target/pi/dense_2/kernel*
dtype0*
_output_shapes
:
�
7target/pi/dense_2/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *�_��*+
_class!
loc:@target/pi/dense_2/kernel*
dtype0
�
7target/pi/dense_2/kernel/Initializer/random_uniform/maxConst*
_output_shapes
: *
valueB
 *�_�>*+
_class!
loc:@target/pi/dense_2/kernel*
dtype0
�
Atarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/pi/dense_2/kernel/Initializer/random_uniform/shape*

seed*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
seed2�*
dtype0*
_output_shapes

:@
�
7target/pi/dense_2/kernel/Initializer/random_uniform/subSub7target/pi/dense_2/kernel/Initializer/random_uniform/max7target/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes
: 
�
7target/pi/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/pi/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/pi/dense_2/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes

:@
�
3target/pi/dense_2/kernel/Initializer/random_uniformAdd7target/pi/dense_2/kernel/Initializer/random_uniform/mul7target/pi/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
_output_shapes

:@
�
target/pi/dense_2/kernel
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *+
_class!
loc:@target/pi/dense_2/kernel
�
target/pi/dense_2/kernel/AssignAssigntarget/pi/dense_2/kernel3target/pi/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
target/pi/dense_2/kernel/readIdentitytarget/pi/dense_2/kernel*
_output_shapes

:@*
T0*+
_class!
loc:@target/pi/dense_2/kernel
�
(target/pi/dense_2/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/pi/dense_2/bias*
dtype0*
_output_shapes
:
�
target/pi/dense_2/bias
VariableV2*
_output_shapes
:*
shared_name *)
_class
loc:@target/pi/dense_2/bias*
	container *
shape:*
dtype0
�
target/pi/dense_2/bias/AssignAssigntarget/pi/dense_2/bias(target/pi/dense_2/bias/Initializer/zeros*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
target/pi/dense_2/bias/readIdentitytarget/pi/dense_2/bias*
_output_shapes
:*
T0*)
_class
loc:@target/pi/dense_2/bias
�
target/pi/dense_2/MatMulMatMultarget/pi/dense_1/Relutarget/pi/dense_2/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
target/pi/dense_2/BiasAddBiasAddtarget/pi/dense_2/MatMultarget/pi/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
i
target/pi/SoftmaxSoftmaxtarget/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:���������
o
target/pi/LogSoftmax
LogSoftmaxtarget/pi/dense_2/BiasAdd*
T0*'
_output_shapes
:���������
e
target/pi/ArgMax/dimensionConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
target/pi/ArgMaxArgMaxtarget/pi/dense_2/BiasAddtarget/pi/ArgMax/dimension*#
_output_shapes
:���������*

Tidx0*
T0*
output_type0	
s
target/pi/Categorical/probsSoftmaxtarget/pi/dense_2/BiasAdd*'
_output_shapes
:���������*
T0
b
 target/pi/Categorical/batch_rankConst*
value	B :*
dtype0*
_output_shapes
: 
{
"target/pi/Categorical/logits_shapeShapetarget/pi/dense_2/BiasAdd*
_output_shapes
:*
T0*
out_type0
b
 target/pi/Categorical/event_sizeConst*
value	B :*
dtype0*
_output_shapes
: 

5target/pi/Categorical/batch_shape/strided_slice/stackConst*
valueB: *
dtype0*
_output_shapes
:
�
7target/pi/Categorical/batch_shape/strided_slice/stack_1Const*
valueB:
���������*
dtype0*
_output_shapes
:
�
7target/pi/Categorical/batch_shape/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
/target/pi/Categorical/batch_shape/strided_sliceStridedSlice"target/pi/Categorical/logits_shape5target/pi/Categorical/batch_shape/strided_slice/stack7target/pi/Categorical/batch_shape/strided_slice/stack_17target/pi/Categorical/batch_shape/strided_slice/stack_2*
Index0*
T0*
shrink_axis_mask *

begin_mask*
ellipsis_mask *
new_axis_mask *
end_mask *
_output_shapes
:
l
)target/pi/Categorical/sample/sample_shapeConst*
valueB *
dtype0*
_output_shapes
: 
�
@target/pi/Categorical/sample/multinomial/Multinomial/num_samplesConst*
value	B :*
dtype0*
_output_shapes
: 
�
4target/pi/Categorical/sample/multinomial/MultinomialMultinomialtarget/pi/dense_2/BiasAdd@target/pi/Categorical/sample/multinomial/Multinomial/num_samples*
T0*'
_output_shapes
:���������*
seed2�*

seed*
output_dtype0
|
+target/pi/Categorical/sample/transpose/permConst*
_output_shapes
:*
valueB"       *
dtype0
�
&target/pi/Categorical/sample/transpose	Transpose4target/pi/Categorical/sample/multinomial/Multinomial+target/pi/Categorical/sample/transpose/perm*'
_output_shapes
:���������*
Tperm0*
T0
�
1target/pi/Categorical/batch_shape_tensor/IdentityIdentity/target/pi/Categorical/batch_shape/strided_slice*
T0*
_output_shapes
:
v
,target/pi/Categorical/sample/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
j
(target/pi/Categorical/sample/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
#target/pi/Categorical/sample/concatConcatV2,target/pi/Categorical/sample/concat/values_01target/pi/Categorical/batch_shape_tensor/Identity(target/pi/Categorical/sample/concat/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
$target/pi/Categorical/sample/ReshapeReshape&target/pi/Categorical/sample/transpose#target/pi/Categorical/sample/concat*'
_output_shapes
:���������*
T0*
Tshape0
�
"target/pi/Categorical/sample/ShapeShape$target/pi/Categorical/sample/Reshape*
T0*
out_type0*
_output_shapes
:
z
0target/pi/Categorical/sample/strided_slice/stackConst*
dtype0*
_output_shapes
:*
valueB:
|
2target/pi/Categorical/sample/strided_slice/stack_1Const*
valueB: *
dtype0*
_output_shapes
:
|
2target/pi/Categorical/sample/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
*target/pi/Categorical/sample/strided_sliceStridedSlice"target/pi/Categorical/sample/Shape0target/pi/Categorical/sample/strided_slice/stack2target/pi/Categorical/sample/strided_slice/stack_12target/pi/Categorical/sample/strided_slice/stack_2*
shrink_axis_mask *
ellipsis_mask *

begin_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
l
*target/pi/Categorical/sample/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
%target/pi/Categorical/sample/concat_1ConcatV2)target/pi/Categorical/sample/sample_shape*target/pi/Categorical/sample/strided_slice*target/pi/Categorical/sample/concat_1/axis*

Tidx0*
T0*
N*
_output_shapes
:
�
&target/pi/Categorical/sample/Reshape_1Reshape$target/pi/Categorical/sample/Reshape%target/pi/Categorical/sample/concat_1*
T0*
Tshape0*#
_output_shapes
:���������
�
7target/q1/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *)
_class
loc:@target/q1/dense/kernel*
dtype0*
_output_shapes
:
�
5target/q1/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *0��*)
_class
loc:@target/q1/dense/kernel*
dtype0*
_output_shapes
: 
�
5target/q1/dense/kernel/Initializer/random_uniform/maxConst*
dtype0*
_output_shapes
: *
valueB
 *0�>*)
_class
loc:@target/q1/dense/kernel
�
?target/q1/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7target/q1/dense/kernel/Initializer/random_uniform/shape*)
_class
loc:@target/q1/dense/kernel*
seed2�*
dtype0*
_output_shapes

:@*

seed*
T0
�
5target/q1/dense/kernel/Initializer/random_uniform/subSub5target/q1/dense/kernel/Initializer/random_uniform/max5target/q1/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@target/q1/dense/kernel*
_output_shapes
: 
�
5target/q1/dense/kernel/Initializer/random_uniform/mulMul?target/q1/dense/kernel/Initializer/random_uniform/RandomUniform5target/q1/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@target/q1/dense/kernel*
_output_shapes

:@
�
1target/q1/dense/kernel/Initializer/random_uniformAdd5target/q1/dense/kernel/Initializer/random_uniform/mul5target/q1/dense/kernel/Initializer/random_uniform/min*)
_class
loc:@target/q1/dense/kernel*
_output_shapes

:@*
T0
�
target/q1/dense/kernel
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *)
_class
loc:@target/q1/dense/kernel
�
target/q1/dense/kernel/AssignAssigntarget/q1/dense/kernel1target/q1/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@target/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
target/q1/dense/kernel/readIdentitytarget/q1/dense/kernel*
T0*)
_class
loc:@target/q1/dense/kernel*
_output_shapes

:@
�
&target/q1/dense/bias/Initializer/zerosConst*
valueB@*    *'
_class
loc:@target/q1/dense/bias*
dtype0*
_output_shapes
:@
�
target/q1/dense/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@target/q1/dense/bias*
	container *
shape:@
�
target/q1/dense/bias/AssignAssigntarget/q1/dense/bias&target/q1/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes
:@
�
target/q1/dense/bias/readIdentitytarget/q1/dense/bias*
T0*'
_class
loc:@target/q1/dense/bias*
_output_shapes
:@
�
target/q1/dense/MatMulMatMulPlaceholder_2target/q1/dense/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
target/q1/dense/BiasAddBiasAddtarget/q1/dense/MatMultarget/q1/dense/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������@
g
target/q1/dense/ReluRelutarget/q1/dense/BiasAdd*
T0*'
_output_shapes
:���������@
�
9target/q1/dense_1/kernel/Initializer/random_uniform/shapeConst*
valueB"@   @   *+
_class!
loc:@target/q1/dense_1/kernel*
dtype0*
_output_shapes
:
�
7target/q1/dense_1/kernel/Initializer/random_uniform/minConst*
_output_shapes
: *
valueB
 *׳]�*+
_class!
loc:@target/q1/dense_1/kernel*
dtype0
�
7target/q1/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*+
_class!
loc:@target/q1/dense_1/kernel*
dtype0*
_output_shapes
: 
�
Atarget/q1/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q1/dense_1/kernel/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes

:@@*

seed*
T0*+
_class!
loc:@target/q1/dense_1/kernel
�
7target/q1/dense_1/kernel/Initializer/random_uniform/subSub7target/q1/dense_1/kernel/Initializer/random_uniform/max7target/q1/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
_output_shapes
: 
�
7target/q1/dense_1/kernel/Initializer/random_uniform/mulMulAtarget/q1/dense_1/kernel/Initializer/random_uniform/RandomUniform7target/q1/dense_1/kernel/Initializer/random_uniform/sub*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
_output_shapes

:@@
�
3target/q1/dense_1/kernel/Initializer/random_uniformAdd7target/q1/dense_1/kernel/Initializer/random_uniform/mul7target/q1/dense_1/kernel/Initializer/random_uniform/min*
_output_shapes

:@@*
T0*+
_class!
loc:@target/q1/dense_1/kernel
�
target/q1/dense_1/kernel
VariableV2*
shared_name *+
_class!
loc:@target/q1/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@
�
target/q1/dense_1/kernel/AssignAssigntarget/q1/dense_1/kernel3target/q1/dense_1/kernel/Initializer/random_uniform*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel
�
target/q1/dense_1/kernel/readIdentitytarget/q1/dense_1/kernel*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
_output_shapes

:@@
�
(target/q1/dense_1/bias/Initializer/zerosConst*
valueB@*    *)
_class
loc:@target/q1/dense_1/bias*
dtype0*
_output_shapes
:@
�
target/q1/dense_1/bias
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *)
_class
loc:@target/q1/dense_1/bias*
	container *
shape:@
�
target/q1/dense_1/bias/AssignAssigntarget/q1/dense_1/bias(target/q1/dense_1/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
target/q1/dense_1/bias/readIdentitytarget/q1/dense_1/bias*
_output_shapes
:@*
T0*)
_class
loc:@target/q1/dense_1/bias
�
target/q1/dense_1/MatMulMatMultarget/q1/dense/Relutarget/q1/dense_1/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( 
�
target/q1/dense_1/BiasAddBiasAddtarget/q1/dense_1/MatMultarget/q1/dense_1/bias/read*
data_formatNHWC*'
_output_shapes
:���������@*
T0
k
target/q1/dense_1/ReluRelutarget/q1/dense_1/BiasAdd*
T0*'
_output_shapes
:���������@
�
9target/q1/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *+
_class!
loc:@target/q1/dense_2/kernel*
dtype0*
_output_shapes
:
�
7target/q1/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *�_��*+
_class!
loc:@target/q1/dense_2/kernel*
dtype0*
_output_shapes
: 
�
7target/q1/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *�_�>*+
_class!
loc:@target/q1/dense_2/kernel*
dtype0*
_output_shapes
: 
�
Atarget/q1/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q1/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
seed2�
�
7target/q1/dense_2/kernel/Initializer/random_uniform/subSub7target/q1/dense_2/kernel/Initializer/random_uniform/max7target/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
_output_shapes
: 
�
7target/q1/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/q1/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/q1/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*+
_class!
loc:@target/q1/dense_2/kernel
�
3target/q1/dense_2/kernel/Initializer/random_uniformAdd7target/q1/dense_2/kernel/Initializer/random_uniform/mul7target/q1/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
_output_shapes

:@
�
target/q1/dense_2/kernel
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *+
_class!
loc:@target/q1/dense_2/kernel
�
target/q1/dense_2/kernel/AssignAssigntarget/q1/dense_2/kernel3target/q1/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
target/q1/dense_2/kernel/readIdentitytarget/q1/dense_2/kernel*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
_output_shapes

:@
�
(target/q1/dense_2/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/q1/dense_2/bias*
dtype0*
_output_shapes
:
�
target/q1/dense_2/bias
VariableV2*
	container *
shape:*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@target/q1/dense_2/bias
�
target/q1/dense_2/bias/AssignAssigntarget/q1/dense_2/bias(target/q1/dense_2/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
�
target/q1/dense_2/bias/readIdentitytarget/q1/dense_2/bias*
T0*)
_class
loc:@target/q1/dense_2/bias*
_output_shapes
:
�
target/q1/dense_2/MatMulMatMultarget/q1/dense_1/Relutarget/q1/dense_2/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b( 
�
target/q1/dense_2/BiasAddBiasAddtarget/q1/dense_2/MatMultarget/q1/dense_2/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������
�
7target/q2/dense/kernel/Initializer/random_uniform/shapeConst*
valueB"   @   *)
_class
loc:@target/q2/dense/kernel*
dtype0*
_output_shapes
:
�
5target/q2/dense/kernel/Initializer/random_uniform/minConst*
valueB
 *0��*)
_class
loc:@target/q2/dense/kernel*
dtype0*
_output_shapes
: 
�
5target/q2/dense/kernel/Initializer/random_uniform/maxConst*
valueB
 *0�>*)
_class
loc:@target/q2/dense/kernel*
dtype0*
_output_shapes
: 
�
?target/q2/dense/kernel/Initializer/random_uniform/RandomUniformRandomUniform7target/q2/dense/kernel/Initializer/random_uniform/shape*
seed2�*
dtype0*
_output_shapes

:@*

seed*
T0*)
_class
loc:@target/q2/dense/kernel
�
5target/q2/dense/kernel/Initializer/random_uniform/subSub5target/q2/dense/kernel/Initializer/random_uniform/max5target/q2/dense/kernel/Initializer/random_uniform/min*)
_class
loc:@target/q2/dense/kernel*
_output_shapes
: *
T0
�
5target/q2/dense/kernel/Initializer/random_uniform/mulMul?target/q2/dense/kernel/Initializer/random_uniform/RandomUniform5target/q2/dense/kernel/Initializer/random_uniform/sub*
T0*)
_class
loc:@target/q2/dense/kernel*
_output_shapes

:@
�
1target/q2/dense/kernel/Initializer/random_uniformAdd5target/q2/dense/kernel/Initializer/random_uniform/mul5target/q2/dense/kernel/Initializer/random_uniform/min*
T0*)
_class
loc:@target/q2/dense/kernel*
_output_shapes

:@
�
target/q2/dense/kernel
VariableV2*
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *)
_class
loc:@target/q2/dense/kernel*
	container 
�
target/q2/dense/kernel/AssignAssigntarget/q2/dense/kernel1target/q2/dense/kernel/Initializer/random_uniform*
use_locking(*
T0*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
target/q2/dense/kernel/readIdentitytarget/q2/dense/kernel*
T0*)
_class
loc:@target/q2/dense/kernel*
_output_shapes

:@
�
&target/q2/dense/bias/Initializer/zerosConst*
valueB@*    *'
_class
loc:@target/q2/dense/bias*
dtype0*
_output_shapes
:@
�
target/q2/dense/bias
VariableV2*
shared_name *'
_class
loc:@target/q2/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
target/q2/dense/bias/AssignAssigntarget/q2/dense/bias&target/q2/dense/bias/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
target/q2/dense/bias/readIdentitytarget/q2/dense/bias*
T0*'
_class
loc:@target/q2/dense/bias*
_output_shapes
:@
�
target/q2/dense/MatMulMatMulPlaceholder_2target/q2/dense/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
target/q2/dense/BiasAddBiasAddtarget/q2/dense/MatMultarget/q2/dense/bias/read*
data_formatNHWC*'
_output_shapes
:���������@*
T0
g
target/q2/dense/ReluRelutarget/q2/dense/BiasAdd*'
_output_shapes
:���������@*
T0
�
9target/q2/dense_1/kernel/Initializer/random_uniform/shapeConst*
_output_shapes
:*
valueB"@   @   *+
_class!
loc:@target/q2/dense_1/kernel*
dtype0
�
7target/q2/dense_1/kernel/Initializer/random_uniform/minConst*
valueB
 *׳]�*+
_class!
loc:@target/q2/dense_1/kernel*
dtype0*
_output_shapes
: 
�
7target/q2/dense_1/kernel/Initializer/random_uniform/maxConst*
valueB
 *׳]>*+
_class!
loc:@target/q2/dense_1/kernel*
dtype0*
_output_shapes
: 
�
Atarget/q2/dense_1/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q2/dense_1/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@@*

seed*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
seed2�
�
7target/q2/dense_1/kernel/Initializer/random_uniform/subSub7target/q2/dense_1/kernel/Initializer/random_uniform/max7target/q2/dense_1/kernel/Initializer/random_uniform/min*+
_class!
loc:@target/q2/dense_1/kernel*
_output_shapes
: *
T0
�
7target/q2/dense_1/kernel/Initializer/random_uniform/mulMulAtarget/q2/dense_1/kernel/Initializer/random_uniform/RandomUniform7target/q2/dense_1/kernel/Initializer/random_uniform/sub*
_output_shapes

:@@*
T0*+
_class!
loc:@target/q2/dense_1/kernel
�
3target/q2/dense_1/kernel/Initializer/random_uniformAdd7target/q2/dense_1/kernel/Initializer/random_uniform/mul7target/q2/dense_1/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
_output_shapes

:@@
�
target/q2/dense_1/kernel
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *+
_class!
loc:@target/q2/dense_1/kernel*
	container *
shape
:@@
�
target/q2/dense_1/kernel/AssignAssigntarget/q2/dense_1/kernel3target/q2/dense_1/kernel/Initializer/random_uniform*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
target/q2/dense_1/kernel/readIdentitytarget/q2/dense_1/kernel*
_output_shapes

:@@*
T0*+
_class!
loc:@target/q2/dense_1/kernel
�
(target/q2/dense_1/bias/Initializer/zerosConst*
_output_shapes
:@*
valueB@*    *)
_class
loc:@target/q2/dense_1/bias*
dtype0
�
target/q2/dense_1/bias
VariableV2*
shared_name *)
_class
loc:@target/q2/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
target/q2/dense_1/bias/AssignAssigntarget/q2/dense_1/bias(target/q2/dense_1/bias/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@target/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
target/q2/dense_1/bias/readIdentitytarget/q2/dense_1/bias*)
_class
loc:@target/q2/dense_1/bias*
_output_shapes
:@*
T0
�
target/q2/dense_1/MatMulMatMultarget/q2/dense/Relutarget/q2/dense_1/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b( *
T0
�
target/q2/dense_1/BiasAddBiasAddtarget/q2/dense_1/MatMultarget/q2/dense_1/bias/read*
T0*
data_formatNHWC*'
_output_shapes
:���������@
k
target/q2/dense_1/ReluRelutarget/q2/dense_1/BiasAdd*
T0*'
_output_shapes
:���������@
�
9target/q2/dense_2/kernel/Initializer/random_uniform/shapeConst*
valueB"@      *+
_class!
loc:@target/q2/dense_2/kernel*
dtype0*
_output_shapes
:
�
7target/q2/dense_2/kernel/Initializer/random_uniform/minConst*
valueB
 *�_��*+
_class!
loc:@target/q2/dense_2/kernel*
dtype0*
_output_shapes
: 
�
7target/q2/dense_2/kernel/Initializer/random_uniform/maxConst*
valueB
 *�_�>*+
_class!
loc:@target/q2/dense_2/kernel*
dtype0*
_output_shapes
: 
�
Atarget/q2/dense_2/kernel/Initializer/random_uniform/RandomUniformRandomUniform9target/q2/dense_2/kernel/Initializer/random_uniform/shape*
dtype0*
_output_shapes

:@*

seed*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
seed2�
�
7target/q2/dense_2/kernel/Initializer/random_uniform/subSub7target/q2/dense_2/kernel/Initializer/random_uniform/max7target/q2/dense_2/kernel/Initializer/random_uniform/min*
_output_shapes
: *
T0*+
_class!
loc:@target/q2/dense_2/kernel
�
7target/q2/dense_2/kernel/Initializer/random_uniform/mulMulAtarget/q2/dense_2/kernel/Initializer/random_uniform/RandomUniform7target/q2/dense_2/kernel/Initializer/random_uniform/sub*
_output_shapes

:@*
T0*+
_class!
loc:@target/q2/dense_2/kernel
�
3target/q2/dense_2/kernel/Initializer/random_uniformAdd7target/q2/dense_2/kernel/Initializer/random_uniform/mul7target/q2/dense_2/kernel/Initializer/random_uniform/min*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
_output_shapes

:@
�
target/q2/dense_2/kernel
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *+
_class!
loc:@target/q2/dense_2/kernel
�
target/q2/dense_2/kernel/AssignAssigntarget/q2/dense_2/kernel3target/q2/dense_2/kernel/Initializer/random_uniform*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
target/q2/dense_2/kernel/readIdentitytarget/q2/dense_2/kernel*
_output_shapes

:@*
T0*+
_class!
loc:@target/q2/dense_2/kernel
�
(target/q2/dense_2/bias/Initializer/zerosConst*
valueB*    *)
_class
loc:@target/q2/dense_2/bias*
dtype0*
_output_shapes
:
�
target/q2/dense_2/bias
VariableV2*
dtype0*
_output_shapes
:*
shared_name *)
_class
loc:@target/q2/dense_2/bias*
	container *
shape:
�
target/q2/dense_2/bias/AssignAssigntarget/q2/dense_2/bias(target/q2/dense_2/bias/Initializer/zeros*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
target/q2/dense_2/bias/readIdentitytarget/q2/dense_2/bias*
_output_shapes
:*
T0*)
_class
loc:@target/q2/dense_2/bias
�
target/q2/dense_2/MatMulMatMultarget/q2/dense_1/Relutarget/q2/dense_2/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b( *
T0
�
target/q2/dense_2/BiasAddBiasAddtarget/q2/dense_2/MatMultarget/q2/dense_2/bias/read*
data_formatNHWC*'
_output_shapes
:���������*
T0
p
addAddV2main/q1/dense_2/BiasAddmain/q2/dense_2/BiasAdd*
T0*'
_output_shapes
:���������
N
	truediv/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
T
truedivRealDivadd	truediv/y*'
_output_shapes
:���������*
T0
n
subSubmain/q1/dense_2/BiasAddmain/q2/dense_2/BiasAdd*
T0*'
_output_shapes
:���������
A
AbsAbssub*
T0*'
_output_shapes
:���������
P
truediv_1/yConst*
_output_shapes
: *
valueB
 *   @*
dtype0
X
	truediv_1RealDivAbstruediv_1/y*'
_output_shapes
:���������*
T0
L
mul_1/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
R
mul_1Mulmul_1/x	truediv_1*
T0*'
_output_shapes
:���������
P
add_1AddV2truedivmul_1*
T0*'
_output_shapes
:���������
L
mul_2/xConst*
valueB
 *  ��*
dtype0*
_output_shapes
: 
R
mul_2Mulmul_2/x	truediv_1*
T0*'
_output_shapes
:���������
P
add_2AddV2truedivmul_2*'
_output_shapes
:���������*
T0
K
SoftmaxSoftmaxadd_1*
T0*'
_output_shapes
:���������
�
!R/Initializer/random_normal/shapeConst*
valueB"      *
_class

loc:@R*
dtype0*
_output_shapes
:
{
 R/Initializer/random_normal/meanConst*
_output_shapes
: *
valueB
 *    *
_class

loc:@R*
dtype0
}
"R/Initializer/random_normal/stddevConst*
valueB
 *
�#<*
_class

loc:@R*
dtype0*
_output_shapes
: 
�
0R/Initializer/random_normal/RandomStandardNormalRandomStandardNormal!R/Initializer/random_normal/shape*
seed2�*
dtype0*
_output_shapes

:*

seed*
T0*
_class

loc:@R
�
R/Initializer/random_normal/mulMul0R/Initializer/random_normal/RandomStandardNormal"R/Initializer/random_normal/stddev*
_class

loc:@R*
_output_shapes

:*
T0
�
R/Initializer/random_normalAddR/Initializer/random_normal/mul R/Initializer/random_normal/mean*
_output_shapes

:*
T0*
_class

loc:@R
�
R
VariableV2*
_class

loc:@R*
	container *
shape
:*
dtype0*
_output_shapes

:*
shared_name 
�
R/AssignAssignRR/Initializer/random_normal*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes

:
T
R/readIdentityR*
T0*
_class

loc:@R*
_output_shapes

:
�
AssignAssignRmain/pi/dense_2/BiasAdd*
validate_shape(*
_output_shapes

:*
use_locking( *
T0*
_class

loc:@R
E
	Softmax_1SoftmaxR/read*
_output_shapes

:*
T0
R
Mul_3Mul	Softmax_1Softmax*
T0*'
_output_shapes
:���������
X
Const_1Const*
valueB"       *
dtype0*
_output_shapes
:
X
SumSumMul_3Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
b
	truediv_2RealDiv	Softmax_1main/pi/Softmax*'
_output_shapes
:���������*
T0
I
Log_1Log	truediv_2*
T0*'
_output_shapes
:���������
P
Mul_4Mul	Softmax_1Log_1*
T0*'
_output_shapes
:���������
X
Const_2Const*
dtype0*
_output_shapes
:*
valueB"       
Z
Sum_1SumMul_4Const_2*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
L
sub_1/yConst*
valueB
 *���=*
dtype0*
_output_shapes
: 
=
sub_1SubSum_1sub_1/y*
_output_shapes
: *
T0
4
ReluRelusub_1*
T0*
_output_shapes
: 
0
NegNegSum*
_output_shapes
: *
T0
:
add_3AddV2NegRelu*
T0*
_output_shapes
: 
R
gradients/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
X
gradients/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
o
gradients/FillFillgradients/Shapegradients/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
>
%gradients/add_3_grad/tuple/group_depsNoOp^gradients/Fill
�
-gradients/add_3_grad/tuple/control_dependencyIdentitygradients/Fill&^gradients/add_3_grad/tuple/group_deps*
_output_shapes
: *
T0*!
_class
loc:@gradients/Fill
�
/gradients/add_3_grad/tuple/control_dependency_1Identitygradients/Fill&^gradients/add_3_grad/tuple/group_deps*
T0*!
_class
loc:@gradients/Fill*
_output_shapes
: 
m
gradients/Neg_grad/NegNeg-gradients/add_3_grad/tuple/control_dependency*
T0*
_output_shapes
: 
�
gradients/Relu_grad/ReluGradReluGrad/gradients/add_3_grad/tuple/control_dependency_1Relu*
T0*
_output_shapes
: 
q
 gradients/Sum_grad/Reshape/shapeConst*
dtype0*
_output_shapes
:*
valueB"      
�
gradients/Sum_grad/ReshapeReshapegradients/Neg_grad/Neg gradients/Sum_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
]
gradients/Sum_grad/ShapeShapeMul_3*
_output_shapes
:*
T0*
out_type0
�
gradients/Sum_grad/TileTilegradients/Sum_grad/Reshapegradients/Sum_grad/Shape*
T0*'
_output_shapes
:���������*

Tmultiples0
^
gradients/sub_1_grad/NegNeggradients/Relu_grad/ReluGrad*
T0*
_output_shapes
: 
g
%gradients/sub_1_grad/tuple/group_depsNoOp^gradients/Relu_grad/ReluGrad^gradients/sub_1_grad/Neg
�
-gradients/sub_1_grad/tuple/control_dependencyIdentitygradients/Relu_grad/ReluGrad&^gradients/sub_1_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Relu_grad/ReluGrad*
_output_shapes
: 
�
/gradients/sub_1_grad/tuple/control_dependency_1Identitygradients/sub_1_grad/Neg&^gradients/sub_1_grad/tuple/group_deps*
T0*+
_class!
loc:@gradients/sub_1_grad/Neg*
_output_shapes
: 
c
gradients/Mul_3_grad/ShapeShape	Softmax_1*
T0*
out_type0*
_output_shapes
:
c
gradients/Mul_3_grad/Shape_1ShapeSoftmax*
T0*
out_type0*
_output_shapes
:
�
*gradients/Mul_3_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_3_grad/Shapegradients/Mul_3_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
s
gradients/Mul_3_grad/MulMulgradients/Sum_grad/TileSoftmax*
T0*'
_output_shapes
:���������
�
gradients/Mul_3_grad/SumSumgradients/Mul_3_grad/Mul*gradients/Mul_3_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Mul_3_grad/ReshapeReshapegradients/Mul_3_grad/Sumgradients/Mul_3_grad/Shape*
T0*
Tshape0*
_output_shapes

:
w
gradients/Mul_3_grad/Mul_1Mul	Softmax_1gradients/Sum_grad/Tile*
T0*'
_output_shapes
:���������
�
gradients/Mul_3_grad/Sum_1Sumgradients/Mul_3_grad/Mul_1,gradients/Mul_3_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients/Mul_3_grad/Reshape_1Reshapegradients/Mul_3_grad/Sum_1gradients/Mul_3_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
m
%gradients/Mul_3_grad/tuple/group_depsNoOp^gradients/Mul_3_grad/Reshape^gradients/Mul_3_grad/Reshape_1
�
-gradients/Mul_3_grad/tuple/control_dependencyIdentitygradients/Mul_3_grad/Reshape&^gradients/Mul_3_grad/tuple/group_deps*
_output_shapes

:*
T0*/
_class%
#!loc:@gradients/Mul_3_grad/Reshape
�
/gradients/Mul_3_grad/tuple/control_dependency_1Identitygradients/Mul_3_grad/Reshape_1&^gradients/Mul_3_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Mul_3_grad/Reshape_1
s
"gradients/Sum_1_grad/Reshape/shapeConst*
_output_shapes
:*
valueB"      *
dtype0
�
gradients/Sum_1_grad/ReshapeReshape-gradients/sub_1_grad/tuple/control_dependency"gradients/Sum_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes

:
_
gradients/Sum_1_grad/ShapeShapeMul_4*
T0*
out_type0*
_output_shapes
:
�
gradients/Sum_1_grad/TileTilegradients/Sum_1_grad/Reshapegradients/Sum_1_grad/Shape*

Tmultiples0*
T0*'
_output_shapes
:���������
c
gradients/Mul_4_grad/ShapeShape	Softmax_1*
_output_shapes
:*
T0*
out_type0
a
gradients/Mul_4_grad/Shape_1ShapeLog_1*
T0*
out_type0*
_output_shapes
:
�
*gradients/Mul_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/Mul_4_grad/Shapegradients/Mul_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
s
gradients/Mul_4_grad/MulMulgradients/Sum_1_grad/TileLog_1*'
_output_shapes
:���������*
T0
�
gradients/Mul_4_grad/SumSumgradients/Mul_4_grad/Mul*gradients/Mul_4_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients/Mul_4_grad/ReshapeReshapegradients/Mul_4_grad/Sumgradients/Mul_4_grad/Shape*
_output_shapes

:*
T0*
Tshape0
y
gradients/Mul_4_grad/Mul_1Mul	Softmax_1gradients/Sum_1_grad/Tile*'
_output_shapes
:���������*
T0
�
gradients/Mul_4_grad/Sum_1Sumgradients/Mul_4_grad/Mul_1,gradients/Mul_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients/Mul_4_grad/Reshape_1Reshapegradients/Mul_4_grad/Sum_1gradients/Mul_4_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
m
%gradients/Mul_4_grad/tuple/group_depsNoOp^gradients/Mul_4_grad/Reshape^gradients/Mul_4_grad/Reshape_1
�
-gradients/Mul_4_grad/tuple/control_dependencyIdentitygradients/Mul_4_grad/Reshape&^gradients/Mul_4_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients/Mul_4_grad/Reshape*
_output_shapes

:
�
/gradients/Mul_4_grad/tuple/control_dependency_1Identitygradients/Mul_4_grad/Reshape_1&^gradients/Mul_4_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients/Mul_4_grad/Reshape_1
�
gradients/Log_1_grad/Reciprocal
Reciprocal	truediv_20^gradients/Mul_4_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
gradients/Log_1_grad/mulMul/gradients/Mul_4_grad/tuple/control_dependency_1gradients/Log_1_grad/Reciprocal*'
_output_shapes
:���������*
T0
o
gradients/truediv_2_grad/ShapeConst*
_output_shapes
:*
valueB"      *
dtype0
o
 gradients/truediv_2_grad/Shape_1Shapemain/pi/Softmax*
T0*
out_type0*
_output_shapes
:
�
.gradients/truediv_2_grad/BroadcastGradientArgsBroadcastGradientArgsgradients/truediv_2_grad/Shape gradients/truediv_2_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
 gradients/truediv_2_grad/RealDivRealDivgradients/Log_1_grad/mulmain/pi/Softmax*'
_output_shapes
:���������*
T0
�
gradients/truediv_2_grad/SumSum gradients/truediv_2_grad/RealDiv.gradients/truediv_2_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
 gradients/truediv_2_grad/ReshapeReshapegradients/truediv_2_grad/Sumgradients/truediv_2_grad/Shape*
Tshape0*
_output_shapes

:*
T0
W
gradients/truediv_2_grad/NegNeg	Softmax_1*
T0*
_output_shapes

:
�
"gradients/truediv_2_grad/RealDiv_1RealDivgradients/truediv_2_grad/Negmain/pi/Softmax*
T0*'
_output_shapes
:���������
�
"gradients/truediv_2_grad/RealDiv_2RealDiv"gradients/truediv_2_grad/RealDiv_1main/pi/Softmax*
T0*'
_output_shapes
:���������
�
gradients/truediv_2_grad/mulMulgradients/Log_1_grad/mul"gradients/truediv_2_grad/RealDiv_2*
T0*'
_output_shapes
:���������
�
gradients/truediv_2_grad/Sum_1Sumgradients/truediv_2_grad/mul0gradients/truediv_2_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
"gradients/truediv_2_grad/Reshape_1Reshapegradients/truediv_2_grad/Sum_1 gradients/truediv_2_grad/Shape_1*'
_output_shapes
:���������*
T0*
Tshape0
y
)gradients/truediv_2_grad/tuple/group_depsNoOp!^gradients/truediv_2_grad/Reshape#^gradients/truediv_2_grad/Reshape_1
�
1gradients/truediv_2_grad/tuple/control_dependencyIdentity gradients/truediv_2_grad/Reshape*^gradients/truediv_2_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients/truediv_2_grad/Reshape*
_output_shapes

:
�
3gradients/truediv_2_grad/tuple/control_dependency_1Identity"gradients/truediv_2_grad/Reshape_1*^gradients/truediv_2_grad/tuple/group_deps*
T0*5
_class+
)'loc:@gradients/truediv_2_grad/Reshape_1*'
_output_shapes
:���������
�
gradients/AddNAddN-gradients/Mul_3_grad/tuple/control_dependency-gradients/Mul_4_grad/tuple/control_dependency1gradients/truediv_2_grad/tuple/control_dependency*
T0*/
_class%
#!loc:@gradients/Mul_3_grad/Reshape*
N*
_output_shapes

:
g
gradients/Softmax_1_grad/mulMulgradients/AddN	Softmax_1*
T0*
_output_shapes

:
y
.gradients/Softmax_1_grad/Sum/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
gradients/Softmax_1_grad/SumSumgradients/Softmax_1_grad/mul.gradients/Softmax_1_grad/Sum/reduction_indices*
T0*
_output_shapes

:*
	keep_dims(*

Tidx0
z
gradients/Softmax_1_grad/subSubgradients/AddNgradients/Softmax_1_grad/Sum*
T0*
_output_shapes

:
w
gradients/Softmax_1_grad/mul_1Mulgradients/Softmax_1_grad/sub	Softmax_1*
T0*
_output_shapes

:
t
beta1_power/initial_valueConst*
valueB
 *fff?*
_class

loc:@R*
dtype0*
_output_shapes
: 
�
beta1_power
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class

loc:@R*
	container 
�
beta1_power/AssignAssignbeta1_powerbeta1_power/initial_value*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: 
`
beta1_power/readIdentitybeta1_power*
T0*
_class

loc:@R*
_output_shapes
: 
t
beta2_power/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*
_class

loc:@R
�
beta2_power
VariableV2*
dtype0*
_output_shapes
: *
shared_name *
_class

loc:@R*
	container *
shape: 
�
beta2_power/AssignAssignbeta2_powerbeta2_power/initial_value*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: *
use_locking(
`
beta2_power/readIdentitybeta2_power*
T0*
_class

loc:@R*
_output_shapes
: 
�
R/Adam/Initializer/zerosConst*
_class

loc:@R*
valueB*    *
dtype0*
_output_shapes

:
�
R/Adam
VariableV2*
shared_name *
_class

loc:@R*
	container *
shape
:*
dtype0*
_output_shapes

:
�
R/Adam/AssignAssignR/AdamR/Adam/Initializer/zeros*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes

:
^
R/Adam/readIdentityR/Adam*
T0*
_class

loc:@R*
_output_shapes

:
�
R/Adam_1/Initializer/zerosConst*
_output_shapes

:*
_class

loc:@R*
valueB*    *
dtype0
�
R/Adam_1
VariableV2*
dtype0*
_output_shapes

:*
shared_name *
_class

loc:@R*
	container *
shape
:
�
R/Adam_1/AssignAssignR/Adam_1R/Adam_1/Initializer/zeros*
_class

loc:@R*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
b
R/Adam_1/readIdentityR/Adam_1*
T0*
_class

loc:@R*
_output_shapes

:
W
Adam/learning_rateConst*
_output_shapes
: *
valueB
 *��>*
dtype0
O

Adam/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
O

Adam/beta2Const*
valueB
 *w�?*
dtype0*
_output_shapes
: 
Q
Adam/epsilonConst*
dtype0*
_output_shapes
: *
valueB
 *w�+2
�
Adam/update_R/ApplyAdam	ApplyAdamRR/AdamR/Adam_1beta1_power/readbeta2_power/readAdam/learning_rate
Adam/beta1
Adam/beta2Adam/epsilongradients/Softmax_1_grad/mul_1*
_class

loc:@R*
use_nesterov( *
_output_shapes

:*
use_locking( *
T0
~
Adam/mulMulbeta1_power/read
Adam/beta1^Adam/update_R/ApplyAdam*
T0*
_class

loc:@R*
_output_shapes
: 
�
Adam/AssignAssignbeta1_powerAdam/mul*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: *
use_locking( 
�

Adam/mul_1Mulbeta2_power/read
Adam/beta2^Adam/update_R/ApplyAdam*
_output_shapes
: *
T0*
_class

loc:@R
�
Adam/Assign_1Assignbeta2_power
Adam/mul_1*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: *
use_locking( 
D
AdamNoOp^Adam/Assign^Adam/Assign_1^Adam/update_R/ApplyAdam
Q
Categorical/logits/LogLog	Softmax_1*
_output_shapes

:*
T0
X
Categorical/batch_rankConst*
dtype0*
_output_shapes
: *
value	B :
i
Categorical/logits_shapeConst*
_output_shapes
:*
valueB"      *
dtype0
X
Categorical/event_sizeConst*
value	B :*
dtype0*
_output_shapes
: 
a
Categorical/batch_shapeConst*
valueB:*
dtype0*
_output_shapes
:
b
Categorical/sample/sample_shapeConst*
valueB *
dtype0*
_output_shapes
: 
x
6Categorical/sample/multinomial/Multinomial/num_samplesConst*
_output_shapes
: *
value	B :*
dtype0
�
*Categorical/sample/multinomial/MultinomialMultinomialCategorical/logits/Log6Categorical/sample/multinomial/Multinomial/num_samples*
T0*
_output_shapes

:*
seed2�*

seed*
output_dtype0
r
!Categorical/sample/transpose/permConst*
_output_shapes
:*
valueB"       *
dtype0
�
Categorical/sample/transpose	Transpose*Categorical/sample/multinomial/Multinomial!Categorical/sample/transpose/perm*
T0*
_output_shapes

:*
Tperm0
t
*Categorical/batch_shape_tensor/batch_shapeConst*
valueB:*
dtype0*
_output_shapes
:
l
"Categorical/sample/concat/values_0Const*
valueB:*
dtype0*
_output_shapes
:
`
Categorical/sample/concat/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Categorical/sample/concatConcatV2"Categorical/sample/concat/values_0*Categorical/batch_shape_tensor/batch_shapeCategorical/sample/concat/axis*
T0*
N*
_output_shapes
:*

Tidx0
�
Categorical/sample/ReshapeReshapeCategorical/sample/transposeCategorical/sample/concat*
T0*
Tshape0*
_output_shapes

:
i
Categorical/sample/ShapeConst*
dtype0*
_output_shapes
:*
valueB"      
p
&Categorical/sample/strided_slice/stackConst*
valueB:*
dtype0*
_output_shapes
:
r
(Categorical/sample/strided_slice/stack_1Const*
_output_shapes
:*
valueB: *
dtype0
r
(Categorical/sample/strided_slice/stack_2Const*
valueB:*
dtype0*
_output_shapes
:
�
 Categorical/sample/strided_sliceStridedSliceCategorical/sample/Shape&Categorical/sample/strided_slice/stack(Categorical/sample/strided_slice/stack_1(Categorical/sample/strided_slice/stack_2*
shrink_axis_mask *

begin_mask *
ellipsis_mask *
new_axis_mask *
end_mask*
_output_shapes
:*
Index0*
T0
b
 Categorical/sample/concat_1/axisConst*
value	B : *
dtype0*
_output_shapes
: 
�
Categorical/sample/concat_1ConcatV2Categorical/sample/sample_shape Categorical/sample/strided_slice Categorical/sample/concat_1/axis*
N*
_output_shapes
:*

Tidx0*
T0
�
Categorical/sample/Reshape_1ReshapeCategorical/sample/ReshapeCategorical/sample/concat_1*
_output_shapes
:*
T0*
Tshape0
z
MinimumMinimumtarget/q1/dense_2/BiasAddtarget/q2/dense_2/BiasAdd*'
_output_shapes
:���������*
T0
L
sub_2/xConst*
valueB
 *  �?*
dtype0*
_output_shapes
: 
R
sub_2Subsub_2/xPlaceholder_4*#
_output_shapes
:���������*
T0
L
mul_5/xConst*
valueB
 *�p}?*
dtype0*
_output_shapes
: 
J
mul_5Mulmul_5/xsub_2*#
_output_shapes
:���������*
T0
Y
mul_6MulExpmain_1/pi/LogSoftmax*
T0*'
_output_shapes
:���������
N
sub_3SubMinimummul_6*
T0*'
_output_shapes
:���������
X
mul_7Mulmain_1/pi/Softmaxsub_3*
T0*'
_output_shapes
:���������
b
Sum_2/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
w
Sum_2Summul_7Sum_2/reduction_indices*
T0*#
_output_shapes
:���������*
	keep_dims( *

Tidx0
Q
StopGradientStopGradientSum_2*#
_output_shapes
:���������*
T0
O
mul_8Mulmul_5StopGradient*
T0*#
_output_shapes
:���������
R
add_4AddV2Placeholder_3mul_8*
T0*#
_output_shapes
:���������
f
Mul_9Mulmain/q1/dense_2/BiasAddPlaceholder_1*
T0*'
_output_shapes
:���������
Y
Sum_3/reduction_indicesConst*
value	B :*
dtype0*
_output_shapes
: 
w
Sum_3SumMul_9Sum_3/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
g
Mul_10Mulmain/q2/dense_2/BiasAddPlaceholder_1*
T0*'
_output_shapes
:���������
Y
Sum_4/reduction_indicesConst*
_output_shapes
: *
value	B :*
dtype0
x
Sum_4SumMul_10Sum_4/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
H
sub_4Subadd_4Sum_3*
T0*#
_output_shapes
:���������
J
pow/yConst*
dtype0*
_output_shapes
: *
valueB
 *   @
F
powPowsub_4pow/y*#
_output_shapes
:���������*
T0
Q
Const_3Const*
dtype0*
_output_shapes
:*
valueB: 
X
MeanMeanpowConst_3*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
M
mul_11/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
>
mul_11Mulmul_11/xMean*
T0*
_output_shapes
: 
H
sub_5Subadd_4Sum_4*
T0*#
_output_shapes
:���������
L
pow_1/yConst*
valueB
 *   @*
dtype0*
_output_shapes
: 
J
pow_1Powsub_5pow_1/y*
T0*#
_output_shapes
:���������
Q
Const_4Const*
valueB: *
dtype0*
_output_shapes
:
\
Mean_1Meanpow_1Const_4*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
M
mul_12/xConst*
valueB
 *   ?*
dtype0*
_output_shapes
: 
@
mul_12Mulmul_12/xMean_1*
T0*
_output_shapes
: 
?
add_5AddV2mul_11mul_12*
T0*
_output_shapes
: 
X
mul_13MulExpmain/pi/LogSoftmax*
T0*'
_output_shapes
:���������
M
sub_6Submul_13add_2*
T0*'
_output_shapes
:���������
W
mul_14Mulmain/pi/Softmaxsub_6*
T0*'
_output_shapes
:���������
b
Sum_5/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
x
Sum_5Summul_14Sum_5/reduction_indices*#
_output_shapes
:���������*
	keep_dims( *

Tidx0*
T0
Q
Const_5Const*
_output_shapes
:*
valueB: *
dtype0
\
Mean_2MeanSum_5Const_5*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
d
mul_15Mulmain/pi/Softmaxmain/pi/LogSoftmax*'
_output_shapes
:���������*
T0
b
Sum_6/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
x
Sum_6Summul_15Sum_6/reduction_indices*
	keep_dims( *

Tidx0*
T0*#
_output_shapes
:���������
A
Neg_1NegSum_6*
T0*#
_output_shapes
:���������
F
sub_7SubmulNeg_1*
T0*#
_output_shapes
:���������
S
StopGradient_1StopGradientsub_7*
T0*#
_output_shapes
:���������
[
mul_16Mullog_alpha/readStopGradient_1*#
_output_shapes
:���������*
T0
Q
Const_6Const*
valueB: *
dtype0*
_output_shapes
:
]
Mean_3Meanmul_16Const_6*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
5
Neg_2NegMean_3*
T0*
_output_shapes
: 
T
gradients_1/ShapeConst*
valueB *
dtype0*
_output_shapes
: 
Z
gradients_1/grad_ys_0Const*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
gradients_1/FillFillgradients_1/Shapegradients_1/grad_ys_0*
T0*

index_type0*
_output_shapes
: 
o
%gradients_1/Mean_2_grad/Reshape/shapeConst*
valueB:*
dtype0*
_output_shapes
:
�
gradients_1/Mean_2_grad/ReshapeReshapegradients_1/Fill%gradients_1/Mean_2_grad/Reshape/shape*
Tshape0*
_output_shapes
:*
T0
b
gradients_1/Mean_2_grad/ShapeShapeSum_5*
_output_shapes
:*
T0*
out_type0
�
gradients_1/Mean_2_grad/TileTilegradients_1/Mean_2_grad/Reshapegradients_1/Mean_2_grad/Shape*
T0*#
_output_shapes
:���������*

Tmultiples0
d
gradients_1/Mean_2_grad/Shape_1ShapeSum_5*
T0*
out_type0*
_output_shapes
:
b
gradients_1/Mean_2_grad/Shape_2Const*
valueB *
dtype0*
_output_shapes
: 
g
gradients_1/Mean_2_grad/ConstConst*
valueB: *
dtype0*
_output_shapes
:
�
gradients_1/Mean_2_grad/ProdProdgradients_1/Mean_2_grad/Shape_1gradients_1/Mean_2_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
i
gradients_1/Mean_2_grad/Const_1Const*
_output_shapes
:*
valueB: *
dtype0
�
gradients_1/Mean_2_grad/Prod_1Prodgradients_1/Mean_2_grad/Shape_2gradients_1/Mean_2_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
c
!gradients_1/Mean_2_grad/Maximum/yConst*
value	B :*
dtype0*
_output_shapes
: 
�
gradients_1/Mean_2_grad/MaximumMaximumgradients_1/Mean_2_grad/Prod_1!gradients_1/Mean_2_grad/Maximum/y*
T0*
_output_shapes
: 
�
 gradients_1/Mean_2_grad/floordivFloorDivgradients_1/Mean_2_grad/Prodgradients_1/Mean_2_grad/Maximum*
T0*
_output_shapes
: 
�
gradients_1/Mean_2_grad/CastCast gradients_1/Mean_2_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients_1/Mean_2_grad/truedivRealDivgradients_1/Mean_2_grad/Tilegradients_1/Mean_2_grad/Cast*#
_output_shapes
:���������*
T0
b
gradients_1/Sum_5_grad/ShapeShapemul_14*
out_type0*
_output_shapes
:*
T0
�
gradients_1/Sum_5_grad/SizeConst*
_output_shapes
: *
value	B :*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
dtype0
�
gradients_1/Sum_5_grad/addAddV2Sum_5/reduction_indicesgradients_1/Sum_5_grad/Size*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape
�
gradients_1/Sum_5_grad/modFloorModgradients_1/Sum_5_grad/addgradients_1/Sum_5_grad/Size*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
_output_shapes
: 
�
gradients_1/Sum_5_grad/Shape_1Const*
dtype0*
_output_shapes
: *
valueB */
_class%
#!loc:@gradients_1/Sum_5_grad/Shape
�
"gradients_1/Sum_5_grad/range/startConst*
value	B : */
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
dtype0*
_output_shapes
: 
�
"gradients_1/Sum_5_grad/range/deltaConst*
value	B :*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients_1/Sum_5_grad/rangeRange"gradients_1/Sum_5_grad/range/startgradients_1/Sum_5_grad/Size"gradients_1/Sum_5_grad/range/delta*

Tidx0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
_output_shapes
:
�
!gradients_1/Sum_5_grad/Fill/valueConst*
value	B :*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients_1/Sum_5_grad/FillFillgradients_1/Sum_5_grad/Shape_1!gradients_1/Sum_5_grad/Fill/value*
T0*

index_type0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
_output_shapes
: 
�
$gradients_1/Sum_5_grad/DynamicStitchDynamicStitchgradients_1/Sum_5_grad/rangegradients_1/Sum_5_grad/modgradients_1/Sum_5_grad/Shapegradients_1/Sum_5_grad/Fill*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
N*
_output_shapes
:
�
 gradients_1/Sum_5_grad/Maximum/yConst*
value	B :*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients_1/Sum_5_grad/MaximumMaximum$gradients_1/Sum_5_grad/DynamicStitch gradients_1/Sum_5_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
_output_shapes
:
�
gradients_1/Sum_5_grad/floordivFloorDivgradients_1/Sum_5_grad/Shapegradients_1/Sum_5_grad/Maximum*
T0*/
_class%
#!loc:@gradients_1/Sum_5_grad/Shape*
_output_shapes
:
�
gradients_1/Sum_5_grad/ReshapeReshapegradients_1/Mean_2_grad/truediv$gradients_1/Sum_5_grad/DynamicStitch*0
_output_shapes
:������������������*
T0*
Tshape0
�
gradients_1/Sum_5_grad/TileTilegradients_1/Sum_5_grad/Reshapegradients_1/Sum_5_grad/floordiv*'
_output_shapes
:���������*

Tmultiples0*
T0
l
gradients_1/mul_14_grad/ShapeShapemain/pi/Softmax*
T0*
out_type0*
_output_shapes
:
d
gradients_1/mul_14_grad/Shape_1Shapesub_6*
_output_shapes
:*
T0*
out_type0
�
-gradients_1/mul_14_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_14_grad/Shapegradients_1/mul_14_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
x
gradients_1/mul_14_grad/MulMulgradients_1/Sum_5_grad/Tilesub_6*
T0*'
_output_shapes
:���������
�
gradients_1/mul_14_grad/SumSumgradients_1/mul_14_grad/Mul-gradients_1/mul_14_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_1/mul_14_grad/ReshapeReshapegradients_1/mul_14_grad/Sumgradients_1/mul_14_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients_1/mul_14_grad/Mul_1Mulmain/pi/Softmaxgradients_1/Sum_5_grad/Tile*'
_output_shapes
:���������*
T0
�
gradients_1/mul_14_grad/Sum_1Sumgradients_1/mul_14_grad/Mul_1/gradients_1/mul_14_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
!gradients_1/mul_14_grad/Reshape_1Reshapegradients_1/mul_14_grad/Sum_1gradients_1/mul_14_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
v
(gradients_1/mul_14_grad/tuple/group_depsNoOp ^gradients_1/mul_14_grad/Reshape"^gradients_1/mul_14_grad/Reshape_1
�
0gradients_1/mul_14_grad/tuple/control_dependencyIdentitygradients_1/mul_14_grad/Reshape)^gradients_1/mul_14_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_1/mul_14_grad/Reshape*'
_output_shapes
:���������
�
2gradients_1/mul_14_grad/tuple/control_dependency_1Identity!gradients_1/mul_14_grad/Reshape_1)^gradients_1/mul_14_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_1/mul_14_grad/Reshape_1*'
_output_shapes
:���������
�
$gradients_1/main/pi/Softmax_grad/mulMul0gradients_1/mul_14_grad/tuple/control_dependencymain/pi/Softmax*
T0*'
_output_shapes
:���������
�
6gradients_1/main/pi/Softmax_grad/Sum/reduction_indicesConst*
valueB :
���������*
dtype0*
_output_shapes
: 
�
$gradients_1/main/pi/Softmax_grad/SumSum$gradients_1/main/pi/Softmax_grad/mul6gradients_1/main/pi/Softmax_grad/Sum/reduction_indices*
	keep_dims(*

Tidx0*
T0*'
_output_shapes
:���������
�
$gradients_1/main/pi/Softmax_grad/subSub0gradients_1/mul_14_grad/tuple/control_dependency$gradients_1/main/pi/Softmax_grad/Sum*
T0*'
_output_shapes
:���������
�
&gradients_1/main/pi/Softmax_grad/mul_1Mul$gradients_1/main/pi/Softmax_grad/submain/pi/Softmax*
T0*'
_output_shapes
:���������
b
gradients_1/sub_6_grad/ShapeShapemul_13*
T0*
out_type0*
_output_shapes
:
c
gradients_1/sub_6_grad/Shape_1Shapeadd_2*
T0*
out_type0*
_output_shapes
:
�
,gradients_1/sub_6_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/sub_6_grad/Shapegradients_1/sub_6_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_1/sub_6_grad/SumSum2gradients_1/mul_14_grad/tuple/control_dependency_1,gradients_1/sub_6_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients_1/sub_6_grad/ReshapeReshapegradients_1/sub_6_grad/Sumgradients_1/sub_6_grad/Shape*'
_output_shapes
:���������*
T0*
Tshape0
�
gradients_1/sub_6_grad/NegNeg2gradients_1/mul_14_grad/tuple/control_dependency_1*
T0*'
_output_shapes
:���������
�
gradients_1/sub_6_grad/Sum_1Sumgradients_1/sub_6_grad/Neg.gradients_1/sub_6_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
 gradients_1/sub_6_grad/Reshape_1Reshapegradients_1/sub_6_grad/Sum_1gradients_1/sub_6_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
s
'gradients_1/sub_6_grad/tuple/group_depsNoOp^gradients_1/sub_6_grad/Reshape!^gradients_1/sub_6_grad/Reshape_1
�
/gradients_1/sub_6_grad/tuple/control_dependencyIdentitygradients_1/sub_6_grad/Reshape(^gradients_1/sub_6_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_1/sub_6_grad/Reshape*'
_output_shapes
:���������
�
1gradients_1/sub_6_grad/tuple/control_dependency_1Identity gradients_1/sub_6_grad/Reshape_1(^gradients_1/sub_6_grad/tuple/group_deps*3
_class)
'%loc:@gradients_1/sub_6_grad/Reshape_1*'
_output_shapes
:���������*
T0
^
gradients_1/mul_13_grad/ShapeShapeExp*
_output_shapes
: *
T0*
out_type0
q
gradients_1/mul_13_grad/Shape_1Shapemain/pi/LogSoftmax*
out_type0*
_output_shapes
:*
T0
�
-gradients_1/mul_13_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_1/mul_13_grad/Shapegradients_1/mul_13_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_1/mul_13_grad/MulMul/gradients_1/sub_6_grad/tuple/control_dependencymain/pi/LogSoftmax*'
_output_shapes
:���������*
T0
�
gradients_1/mul_13_grad/SumSumgradients_1/mul_13_grad/Mul-gradients_1/mul_13_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients_1/mul_13_grad/ReshapeReshapegradients_1/mul_13_grad/Sumgradients_1/mul_13_grad/Shape*
_output_shapes
: *
T0*
Tshape0
�
gradients_1/mul_13_grad/Mul_1MulExp/gradients_1/sub_6_grad/tuple/control_dependency*'
_output_shapes
:���������*
T0
�
gradients_1/mul_13_grad/Sum_1Sumgradients_1/mul_13_grad/Mul_1/gradients_1/mul_13_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
!gradients_1/mul_13_grad/Reshape_1Reshapegradients_1/mul_13_grad/Sum_1gradients_1/mul_13_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
v
(gradients_1/mul_13_grad/tuple/group_depsNoOp ^gradients_1/mul_13_grad/Reshape"^gradients_1/mul_13_grad/Reshape_1
�
0gradients_1/mul_13_grad/tuple/control_dependencyIdentitygradients_1/mul_13_grad/Reshape)^gradients_1/mul_13_grad/tuple/group_deps*
_output_shapes
: *
T0*2
_class(
&$loc:@gradients_1/mul_13_grad/Reshape
�
2gradients_1/mul_13_grad/tuple/control_dependency_1Identity!gradients_1/mul_13_grad/Reshape_1)^gradients_1/mul_13_grad/tuple/group_deps*4
_class*
(&loc:@gradients_1/mul_13_grad/Reshape_1*'
_output_shapes
:���������*
T0
t
'gradients_1/main/pi/LogSoftmax_grad/ExpExpmain/pi/LogSoftmax*'
_output_shapes
:���������*
T0
�
9gradients_1/main/pi/LogSoftmax_grad/Sum/reduction_indicesConst*
dtype0*
_output_shapes
: *
valueB :
���������
�
'gradients_1/main/pi/LogSoftmax_grad/SumSum2gradients_1/mul_13_grad/tuple/control_dependency_19gradients_1/main/pi/LogSoftmax_grad/Sum/reduction_indices*'
_output_shapes
:���������*
	keep_dims(*

Tidx0*
T0
�
'gradients_1/main/pi/LogSoftmax_grad/mulMul'gradients_1/main/pi/LogSoftmax_grad/Sum'gradients_1/main/pi/LogSoftmax_grad/Exp*
T0*'
_output_shapes
:���������
�
'gradients_1/main/pi/LogSoftmax_grad/subSub2gradients_1/mul_13_grad/tuple/control_dependency_1'gradients_1/main/pi/LogSoftmax_grad/mul*'
_output_shapes
:���������*
T0
�
gradients_1/AddNAddN&gradients_1/main/pi/Softmax_grad/mul_1'gradients_1/main/pi/LogSoftmax_grad/sub*
T0*9
_class/
-+loc:@gradients_1/main/pi/Softmax_grad/mul_1*
N*'
_output_shapes
:���������
�
4gradients_1/main/pi/dense_2/BiasAdd_grad/BiasAddGradBiasAddGradgradients_1/AddN*
T0*
data_formatNHWC*
_output_shapes
:
�
9gradients_1/main/pi/dense_2/BiasAdd_grad/tuple/group_depsNoOp^gradients_1/AddN5^gradients_1/main/pi/dense_2/BiasAdd_grad/BiasAddGrad
�
Agradients_1/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencyIdentitygradients_1/AddN:^gradients_1/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*9
_class/
-+loc:@gradients_1/main/pi/Softmax_grad/mul_1*'
_output_shapes
:���������
�
Cgradients_1/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/pi/dense_2/BiasAdd_grad/BiasAddGrad:^gradients_1/main/pi/dense_2/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/main/pi/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:
�
.gradients_1/main/pi/dense_2/MatMul_grad/MatMulMatMulAgradients_1/main/pi/dense_2/BiasAdd_grad/tuple/control_dependencymain/pi/dense_2/kernel/read*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(*
T0
�
0gradients_1/main/pi/dense_2/MatMul_grad/MatMul_1MatMulmain/pi/dense_1/ReluAgradients_1/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
8gradients_1/main/pi/dense_2/MatMul_grad/tuple/group_depsNoOp/^gradients_1/main/pi/dense_2/MatMul_grad/MatMul1^gradients_1/main/pi/dense_2/MatMul_grad/MatMul_1
�
@gradients_1/main/pi/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/pi/dense_2/MatMul_grad/MatMul9^gradients_1/main/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/pi/dense_2/MatMul_grad/MatMul*'
_output_shapes
:���������@
�
Bgradients_1/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/pi/dense_2/MatMul_grad/MatMul_19^gradients_1/main/pi/dense_2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/main/pi/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
�
.gradients_1/main/pi/dense_1/Relu_grad/ReluGradReluGrad@gradients_1/main/pi/dense_2/MatMul_grad/tuple/control_dependencymain/pi/dense_1/Relu*
T0*'
_output_shapes
:���������@
�
4gradients_1/main/pi/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_1/main/pi/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
9gradients_1/main/pi/dense_1/BiasAdd_grad/tuple/group_depsNoOp5^gradients_1/main/pi/dense_1/BiasAdd_grad/BiasAddGrad/^gradients_1/main/pi/dense_1/Relu_grad/ReluGrad
�
Agradients_1/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients_1/main/pi/dense_1/Relu_grad/ReluGrad:^gradients_1/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/pi/dense_1/Relu_grad/ReluGrad*'
_output_shapes
:���������@
�
Cgradients_1/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_1/main/pi/dense_1/BiasAdd_grad/BiasAddGrad:^gradients_1/main/pi/dense_1/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_1/main/pi/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
.gradients_1/main/pi/dense_1/MatMul_grad/MatMulMatMulAgradients_1/main/pi/dense_1/BiasAdd_grad/tuple/control_dependencymain/pi/dense_1/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(
�
0gradients_1/main/pi/dense_1/MatMul_grad/MatMul_1MatMulmain/pi/dense/ReluAgradients_1/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
�
8gradients_1/main/pi/dense_1/MatMul_grad/tuple/group_depsNoOp/^gradients_1/main/pi/dense_1/MatMul_grad/MatMul1^gradients_1/main/pi/dense_1/MatMul_grad/MatMul_1
�
@gradients_1/main/pi/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_1/main/pi/dense_1/MatMul_grad/MatMul9^gradients_1/main/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_1/main/pi/dense_1/MatMul_grad/MatMul*'
_output_shapes
:���������@
�
Bgradients_1/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_1/main/pi/dense_1/MatMul_grad/MatMul_19^gradients_1/main/pi/dense_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_1/main/pi/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
,gradients_1/main/pi/dense/Relu_grad/ReluGradReluGrad@gradients_1/main/pi/dense_1/MatMul_grad/tuple/control_dependencymain/pi/dense/Relu*
T0*'
_output_shapes
:���������@
�
2gradients_1/main/pi/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_1/main/pi/dense/Relu_grad/ReluGrad*
data_formatNHWC*
_output_shapes
:@*
T0
�
7gradients_1/main/pi/dense/BiasAdd_grad/tuple/group_depsNoOp3^gradients_1/main/pi/dense/BiasAdd_grad/BiasAddGrad-^gradients_1/main/pi/dense/Relu_grad/ReluGrad
�
?gradients_1/main/pi/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_1/main/pi/dense/Relu_grad/ReluGrad8^gradients_1/main/pi/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_1/main/pi/dense/Relu_grad/ReluGrad*'
_output_shapes
:���������@
�
Agradients_1/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_1/main/pi/dense/BiasAdd_grad/BiasAddGrad8^gradients_1/main/pi/dense/BiasAdd_grad/tuple/group_deps*E
_class;
97loc:@gradients_1/main/pi/dense/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@*
T0
�
,gradients_1/main/pi/dense/MatMul_grad/MatMulMatMul?gradients_1/main/pi/dense/BiasAdd_grad/tuple/control_dependencymain/pi/dense/kernel/read*
T0*'
_output_shapes
:���������*
transpose_a( *
transpose_b(
�
.gradients_1/main/pi/dense/MatMul_grad/MatMul_1MatMulPlaceholder?gradients_1/main/pi/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
�
6gradients_1/main/pi/dense/MatMul_grad/tuple/group_depsNoOp-^gradients_1/main/pi/dense/MatMul_grad/MatMul/^gradients_1/main/pi/dense/MatMul_grad/MatMul_1
�
>gradients_1/main/pi/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_1/main/pi/dense/MatMul_grad/MatMul7^gradients_1/main/pi/dense/MatMul_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*?
_class5
31loc:@gradients_1/main/pi/dense/MatMul_grad/MatMul
�
@gradients_1/main/pi/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_1/main/pi/dense/MatMul_grad/MatMul_17^gradients_1/main/pi/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*A
_class7
53loc:@gradients_1/main/pi/dense/MatMul_grad/MatMul_1
�
beta1_power_1/initial_valueConst*
valueB
 *fff?*%
_class
loc:@main/pi/dense/bias*
dtype0*
_output_shapes
: 
�
beta1_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape: 
�
beta1_power_1/AssignAssignbeta1_power_1beta1_power_1/initial_value*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
u
beta1_power_1/readIdentitybeta1_power_1*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
�
beta2_power_1/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*%
_class
loc:@main/pi/dense/bias*
dtype0
�
beta2_power_1
VariableV2*
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape: 
�
beta2_power_1/AssignAssignbeta2_power_1beta2_power_1/initial_value*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
u
beta2_power_1/readIdentitybeta2_power_1*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: *
T0
�
+main/pi/dense/kernel/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/pi/dense/kernel/Adam
VariableV2*
shared_name *'
_class
loc:@main/pi/dense/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
 main/pi/dense/kernel/Adam/AssignAssignmain/pi/dense/kernel/Adam+main/pi/dense/kernel/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
main/pi/dense/kernel/Adam/readIdentitymain/pi/dense/kernel/Adam*
T0*'
_class
loc:@main/pi/dense/kernel*
_output_shapes

:@
�
-main/pi/dense/kernel/Adam_1/Initializer/zerosConst*'
_class
loc:@main/pi/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/pi/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *'
_class
loc:@main/pi/dense/kernel*
	container *
shape
:@
�
"main/pi/dense/kernel/Adam_1/AssignAssignmain/pi/dense/kernel/Adam_1-main/pi/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
 main/pi/dense/kernel/Adam_1/readIdentitymain/pi/dense/kernel/Adam_1*
_output_shapes

:@*
T0*'
_class
loc:@main/pi/dense/kernel
�
)main/pi/dense/bias/Adam/Initializer/zerosConst*%
_class
loc:@main/pi/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/pi/dense/bias/Adam
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container 
�
main/pi/dense/bias/Adam/AssignAssignmain/pi/dense/bias/Adam)main/pi/dense/bias/Adam/Initializer/zeros*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
main/pi/dense/bias/Adam/readIdentitymain/pi/dense/bias/Adam*
_output_shapes
:@*
T0*%
_class
loc:@main/pi/dense/bias
�
+main/pi/dense/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*%
_class
loc:@main/pi/dense/bias*
valueB@*    
�
main/pi/dense/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *%
_class
loc:@main/pi/dense/bias*
	container *
shape:@
�
 main/pi/dense/bias/Adam_1/AssignAssignmain/pi/dense/bias/Adam_1+main/pi/dense/bias/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(
�
main/pi/dense/bias/Adam_1/readIdentitymain/pi/dense/bias/Adam_1*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
:@
�
=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
3main/pi/dense_1/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
-main/pi/dense_1/kernel/Adam/Initializer/zerosFill=main/pi/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/pi/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@main/pi/dense_1/kernel*

index_type0*
_output_shapes

:@@
�
main/pi/dense_1/kernel/Adam
VariableV2*
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *)
_class
loc:@main/pi/dense_1/kernel*
	container 
�
"main/pi/dense_1/kernel/Adam/AssignAssignmain/pi/dense_1/kernel/Adam-main/pi/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
 main/pi/dense_1/kernel/Adam/readIdentitymain/pi/dense_1/kernel/Adam*
T0*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes

:@@
�
?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@main/pi/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/main/pi/dense_1/kernel/Adam_1/Initializer/zerosFill?main/pi/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/pi/dense_1/kernel/Adam_1/Initializer/zeros/Const*
_output_shapes

:@@*
T0*)
_class
loc:@main/pi/dense_1/kernel*

index_type0
�
main/pi/dense_1/kernel/Adam_1
VariableV2*)
_class
loc:@main/pi/dense_1/kernel*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name 
�
$main/pi/dense_1/kernel/Adam_1/AssignAssignmain/pi/dense_1/kernel/Adam_1/main/pi/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
"main/pi/dense_1/kernel/Adam_1/readIdentitymain/pi/dense_1/kernel/Adam_1*)
_class
loc:@main/pi/dense_1/kernel*
_output_shapes

:@@*
T0
�
+main/pi/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/pi/dense_1/bias/Adam
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@main/pi/dense_1/bias*
	container *
shape:@
�
 main/pi/dense_1/bias/Adam/AssignAssignmain/pi/dense_1/bias/Adam+main/pi/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
main/pi/dense_1/bias/Adam/readIdentitymain/pi/dense_1/bias/Adam*
T0*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes
:@
�
-main/pi/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/pi/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/pi/dense_1/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@main/pi/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
"main/pi/dense_1/bias/Adam_1/AssignAssignmain/pi/dense_1/bias/Adam_1-main/pi/dense_1/bias/Adam_1/Initializer/zeros*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
 main/pi/dense_1/bias/Adam_1/readIdentitymain/pi/dense_1/bias/Adam_1*
T0*'
_class
loc:@main/pi/dense_1/bias*
_output_shapes
:@
�
-main/pi/dense_2/kernel/Adam/Initializer/zerosConst*)
_class
loc:@main/pi/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/pi/dense_2/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *)
_class
loc:@main/pi/dense_2/kernel*
	container *
shape
:@
�
"main/pi/dense_2/kernel/Adam/AssignAssignmain/pi/dense_2/kernel/Adam-main/pi/dense_2/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
 main/pi/dense_2/kernel/Adam/readIdentitymain/pi/dense_2/kernel/Adam*
T0*)
_class
loc:@main/pi/dense_2/kernel*
_output_shapes

:@
�
/main/pi/dense_2/kernel/Adam_1/Initializer/zerosConst*)
_class
loc:@main/pi/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/pi/dense_2/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *)
_class
loc:@main/pi/dense_2/kernel*
	container *
shape
:@
�
$main/pi/dense_2/kernel/Adam_1/AssignAssignmain/pi/dense_2/kernel/Adam_1/main/pi/dense_2/kernel/Adam_1/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(
�
"main/pi/dense_2/kernel/Adam_1/readIdentitymain/pi/dense_2/kernel/Adam_1*
_output_shapes

:@*
T0*)
_class
loc:@main/pi/dense_2/kernel
�
+main/pi/dense_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
�
main/pi/dense_2/bias/Adam
VariableV2*
shared_name *'
_class
loc:@main/pi/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
 main/pi/dense_2/bias/Adam/AssignAssignmain/pi/dense_2/bias/Adam+main/pi/dense_2/bias/Adam/Initializer/zeros*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
main/pi/dense_2/bias/Adam/readIdentitymain/pi/dense_2/bias/Adam*
_output_shapes
:*
T0*'
_class
loc:@main/pi/dense_2/bias
�
-main/pi/dense_2/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/pi/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
�
main/pi/dense_2/bias/Adam_1
VariableV2*
shape:*
dtype0*
_output_shapes
:*
shared_name *'
_class
loc:@main/pi/dense_2/bias*
	container 
�
"main/pi/dense_2/bias/Adam_1/AssignAssignmain/pi/dense_2/bias/Adam_1-main/pi/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
 main/pi/dense_2/bias/Adam_1/readIdentitymain/pi/dense_2/bias/Adam_1*
T0*'
_class
loc:@main/pi/dense_2/bias*
_output_shapes
:
Y
Adam_1/learning_rateConst*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
Q
Adam_1/beta1Const*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Q
Adam_1/beta2Const*
_output_shapes
: *
valueB
 *w�?*
dtype0
S
Adam_1/epsilonConst*
valueB
 *��8*
dtype0*
_output_shapes
: 
�
,Adam_1/update_main/pi/dense/kernel/ApplyAdam	ApplyAdammain/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilon@gradients_1/main/pi/dense/MatMul_grad/tuple/control_dependency_1*
T0*'
_class
loc:@main/pi/dense/kernel*
use_nesterov( *
_output_shapes

:@*
use_locking( 
�
*Adam_1/update_main/pi/dense/bias/ApplyAdam	ApplyAdammain/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonAgradients_1/main/pi/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@main/pi/dense/bias*
use_nesterov( *
_output_shapes
:@
�
.Adam_1/update_main/pi/dense_1/kernel/ApplyAdam	ApplyAdammain/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/pi/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@main/pi/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@
�
,Adam_1/update_main/pi/dense_1/bias/ApplyAdam	ApplyAdammain/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/pi/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/pi/dense_1/bias*
use_nesterov( *
_output_shapes
:@
�
.Adam_1/update_main/pi/dense_2/kernel/ApplyAdam	ApplyAdammain/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonBgradients_1/main/pi/dense_2/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@*
use_locking( *
T0*)
_class
loc:@main/pi/dense_2/kernel
�
,Adam_1/update_main/pi/dense_2/bias/ApplyAdam	ApplyAdammain/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1beta1_power_1/readbeta2_power_1/readAdam_1/learning_rateAdam_1/beta1Adam_1/beta2Adam_1/epsilonCgradients_1/main/pi/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/pi/dense_2/bias*
use_nesterov( *
_output_shapes
:
�

Adam_1/mulMulbeta1_power_1/readAdam_1/beta1+^Adam_1/update_main/pi/dense/bias/ApplyAdam-^Adam_1/update_main/pi/dense/kernel/ApplyAdam-^Adam_1/update_main/pi/dense_1/bias/ApplyAdam/^Adam_1/update_main/pi/dense_1/kernel/ApplyAdam-^Adam_1/update_main/pi/dense_2/bias/ApplyAdam/^Adam_1/update_main/pi/dense_2/kernel/ApplyAdam*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
�
Adam_1/AssignAssignbeta1_power_1
Adam_1/mul*
use_locking( *
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
�
Adam_1/mul_1Mulbeta2_power_1/readAdam_1/beta2+^Adam_1/update_main/pi/dense/bias/ApplyAdam-^Adam_1/update_main/pi/dense/kernel/ApplyAdam-^Adam_1/update_main/pi/dense_1/bias/ApplyAdam/^Adam_1/update_main/pi/dense_1/kernel/ApplyAdam-^Adam_1/update_main/pi/dense_2/bias/ApplyAdam/^Adam_1/update_main/pi/dense_2/kernel/ApplyAdam*
T0*%
_class
loc:@main/pi/dense/bias*
_output_shapes
: 
�
Adam_1/Assign_1Assignbeta2_power_1Adam_1/mul_1*
_output_shapes
: *
use_locking( *
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(
�
Adam_1NoOp^Adam_1/Assign^Adam_1/Assign_1+^Adam_1/update_main/pi/dense/bias/ApplyAdam-^Adam_1/update_main/pi/dense/kernel/ApplyAdam-^Adam_1/update_main/pi/dense_1/bias/ApplyAdam/^Adam_1/update_main/pi/dense_1/kernel/ApplyAdam-^Adam_1/update_main/pi/dense_2/bias/ApplyAdam/^Adam_1/update_main/pi/dense_2/kernel/ApplyAdam
]
gradients_2/ShapeConst^Adam_1*
valueB *
dtype0*
_output_shapes
: 
c
gradients_2/grad_ys_0Const^Adam_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
u
gradients_2/FillFillgradients_2/Shapegradients_2/grad_ys_0*
_output_shapes
: *
T0*

index_type0
K
'gradients_2/add_5_grad/tuple/group_depsNoOp^Adam_1^gradients_2/Fill
�
/gradients_2/add_5_grad/tuple/control_dependencyIdentitygradients_2/Fill(^gradients_2/add_5_grad/tuple/group_deps*
T0*#
_class
loc:@gradients_2/Fill*
_output_shapes
: 
�
1gradients_2/add_5_grad/tuple/control_dependency_1Identitygradients_2/Fill(^gradients_2/add_5_grad/tuple/group_deps*
_output_shapes
: *
T0*#
_class
loc:@gradients_2/Fill
z
gradients_2/mul_11_grad/MulMul/gradients_2/add_5_grad/tuple/control_dependencyMean*
_output_shapes
: *
T0
�
gradients_2/mul_11_grad/Mul_1Mul/gradients_2/add_5_grad/tuple/control_dependencymul_11/x*
T0*
_output_shapes
: 
w
(gradients_2/mul_11_grad/tuple/group_depsNoOp^Adam_1^gradients_2/mul_11_grad/Mul^gradients_2/mul_11_grad/Mul_1
�
0gradients_2/mul_11_grad/tuple/control_dependencyIdentitygradients_2/mul_11_grad/Mul)^gradients_2/mul_11_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients_2/mul_11_grad/Mul*
_output_shapes
: 
�
2gradients_2/mul_11_grad/tuple/control_dependency_1Identitygradients_2/mul_11_grad/Mul_1)^gradients_2/mul_11_grad/tuple/group_deps*
T0*0
_class&
$"loc:@gradients_2/mul_11_grad/Mul_1*
_output_shapes
: 
~
gradients_2/mul_12_grad/MulMul1gradients_2/add_5_grad/tuple/control_dependency_1Mean_1*
_output_shapes
: *
T0
�
gradients_2/mul_12_grad/Mul_1Mul1gradients_2/add_5_grad/tuple/control_dependency_1mul_12/x*
T0*
_output_shapes
: 
w
(gradients_2/mul_12_grad/tuple/group_depsNoOp^Adam_1^gradients_2/mul_12_grad/Mul^gradients_2/mul_12_grad/Mul_1
�
0gradients_2/mul_12_grad/tuple/control_dependencyIdentitygradients_2/mul_12_grad/Mul)^gradients_2/mul_12_grad/tuple/group_deps*
T0*.
_class$
" loc:@gradients_2/mul_12_grad/Mul*
_output_shapes
: 
�
2gradients_2/mul_12_grad/tuple/control_dependency_1Identitygradients_2/mul_12_grad/Mul_1)^gradients_2/mul_12_grad/tuple/group_deps*0
_class&
$"loc:@gradients_2/mul_12_grad/Mul_1*
_output_shapes
: *
T0
v
#gradients_2/Mean_grad/Reshape/shapeConst^Adam_1*
valueB:*
dtype0*
_output_shapes
:
�
gradients_2/Mean_grad/ReshapeReshape2gradients_2/mul_11_grad/tuple/control_dependency_1#gradients_2/Mean_grad/Reshape/shape*
_output_shapes
:*
T0*
Tshape0
g
gradients_2/Mean_grad/ShapeShapepow^Adam_1*
T0*
out_type0*
_output_shapes
:
�
gradients_2/Mean_grad/TileTilegradients_2/Mean_grad/Reshapegradients_2/Mean_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
i
gradients_2/Mean_grad/Shape_1Shapepow^Adam_1*
T0*
out_type0*
_output_shapes
:
i
gradients_2/Mean_grad/Shape_2Const^Adam_1*
dtype0*
_output_shapes
: *
valueB 
n
gradients_2/Mean_grad/ConstConst^Adam_1*
valueB: *
dtype0*
_output_shapes
:
�
gradients_2/Mean_grad/ProdProdgradients_2/Mean_grad/Shape_1gradients_2/Mean_grad/Const*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
p
gradients_2/Mean_grad/Const_1Const^Adam_1*
_output_shapes
:*
valueB: *
dtype0
�
gradients_2/Mean_grad/Prod_1Prodgradients_2/Mean_grad/Shape_2gradients_2/Mean_grad/Const_1*
_output_shapes
: *
	keep_dims( *

Tidx0*
T0
j
gradients_2/Mean_grad/Maximum/yConst^Adam_1*
value	B :*
dtype0*
_output_shapes
: 
�
gradients_2/Mean_grad/MaximumMaximumgradients_2/Mean_grad/Prod_1gradients_2/Mean_grad/Maximum/y*
_output_shapes
: *
T0
�
gradients_2/Mean_grad/floordivFloorDivgradients_2/Mean_grad/Prodgradients_2/Mean_grad/Maximum*
T0*
_output_shapes
: 
�
gradients_2/Mean_grad/CastCastgradients_2/Mean_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients_2/Mean_grad/truedivRealDivgradients_2/Mean_grad/Tilegradients_2/Mean_grad/Cast*#
_output_shapes
:���������*
T0
x
%gradients_2/Mean_1_grad/Reshape/shapeConst^Adam_1*
_output_shapes
:*
valueB:*
dtype0
�
gradients_2/Mean_1_grad/ReshapeReshape2gradients_2/mul_12_grad/tuple/control_dependency_1%gradients_2/Mean_1_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
k
gradients_2/Mean_1_grad/ShapeShapepow_1^Adam_1*
T0*
out_type0*
_output_shapes
:
�
gradients_2/Mean_1_grad/TileTilegradients_2/Mean_1_grad/Reshapegradients_2/Mean_1_grad/Shape*

Tmultiples0*
T0*#
_output_shapes
:���������
m
gradients_2/Mean_1_grad/Shape_1Shapepow_1^Adam_1*
_output_shapes
:*
T0*
out_type0
k
gradients_2/Mean_1_grad/Shape_2Const^Adam_1*
dtype0*
_output_shapes
: *
valueB 
p
gradients_2/Mean_1_grad/ConstConst^Adam_1*
valueB: *
dtype0*
_output_shapes
:
�
gradients_2/Mean_1_grad/ProdProdgradients_2/Mean_1_grad/Shape_1gradients_2/Mean_1_grad/Const*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
r
gradients_2/Mean_1_grad/Const_1Const^Adam_1*
valueB: *
dtype0*
_output_shapes
:
�
gradients_2/Mean_1_grad/Prod_1Prodgradients_2/Mean_1_grad/Shape_2gradients_2/Mean_1_grad/Const_1*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
l
!gradients_2/Mean_1_grad/Maximum/yConst^Adam_1*
value	B :*
dtype0*
_output_shapes
: 
�
gradients_2/Mean_1_grad/MaximumMaximumgradients_2/Mean_1_grad/Prod_1!gradients_2/Mean_1_grad/Maximum/y*
T0*
_output_shapes
: 
�
 gradients_2/Mean_1_grad/floordivFloorDivgradients_2/Mean_1_grad/Prodgradients_2/Mean_1_grad/Maximum*
T0*
_output_shapes
: 
�
gradients_2/Mean_1_grad/CastCast gradients_2/Mean_1_grad/floordiv*
Truncate( *
_output_shapes
: *

DstT0*

SrcT0
�
gradients_2/Mean_1_grad/truedivRealDivgradients_2/Mean_1_grad/Tilegradients_2/Mean_1_grad/Cast*
T0*#
_output_shapes
:���������
h
gradients_2/pow_grad/ShapeShapesub_4^Adam_1*
_output_shapes
:*
T0*
out_type0
h
gradients_2/pow_grad/Shape_1Shapepow/y^Adam_1*
_output_shapes
: *
T0*
out_type0
�
*gradients_2/pow_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_grad/Shapegradients_2/pow_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
s
gradients_2/pow_grad/mulMulgradients_2/Mean_grad/truedivpow/y*#
_output_shapes
:���������*
T0
h
gradients_2/pow_grad/sub/yConst^Adam_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
c
gradients_2/pow_grad/subSubpow/ygradients_2/pow_grad/sub/y*
T0*
_output_shapes
: 
n
gradients_2/pow_grad/PowPowsub_4gradients_2/pow_grad/sub*
T0*#
_output_shapes
:���������
�
gradients_2/pow_grad/mul_1Mulgradients_2/pow_grad/mulgradients_2/pow_grad/Pow*
T0*#
_output_shapes
:���������
�
gradients_2/pow_grad/SumSumgradients_2/pow_grad/mul_1*gradients_2/pow_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_2/pow_grad/ReshapeReshapegradients_2/pow_grad/Sumgradients_2/pow_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
l
gradients_2/pow_grad/Greater/yConst^Adam_1*
valueB
 *    *
dtype0*
_output_shapes
: 
|
gradients_2/pow_grad/GreaterGreatersub_4gradients_2/pow_grad/Greater/y*
T0*#
_output_shapes
:���������
r
$gradients_2/pow_grad/ones_like/ShapeShapesub_4^Adam_1*
T0*
out_type0*
_output_shapes
:
r
$gradients_2/pow_grad/ones_like/ConstConst^Adam_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
gradients_2/pow_grad/ones_likeFill$gradients_2/pow_grad/ones_like/Shape$gradients_2/pow_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:���������
�
gradients_2/pow_grad/SelectSelectgradients_2/pow_grad/Greatersub_4gradients_2/pow_grad/ones_like*#
_output_shapes
:���������*
T0
j
gradients_2/pow_grad/LogLoggradients_2/pow_grad/Select*#
_output_shapes
:���������*
T0
j
gradients_2/pow_grad/zeros_like	ZerosLikesub_4^Adam_1*#
_output_shapes
:���������*
T0
�
gradients_2/pow_grad/Select_1Selectgradients_2/pow_grad/Greatergradients_2/pow_grad/Loggradients_2/pow_grad/zeros_like*
T0*#
_output_shapes
:���������
s
gradients_2/pow_grad/mul_2Mulgradients_2/Mean_grad/truedivpow*
T0*#
_output_shapes
:���������
�
gradients_2/pow_grad/mul_3Mulgradients_2/pow_grad/mul_2gradients_2/pow_grad/Select_1*#
_output_shapes
:���������*
T0
�
gradients_2/pow_grad/Sum_1Sumgradients_2/pow_grad/mul_3,gradients_2/pow_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients_2/pow_grad/Reshape_1Reshapegradients_2/pow_grad/Sum_1gradients_2/pow_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
v
%gradients_2/pow_grad/tuple/group_depsNoOp^Adam_1^gradients_2/pow_grad/Reshape^gradients_2/pow_grad/Reshape_1
�
-gradients_2/pow_grad/tuple/control_dependencyIdentitygradients_2/pow_grad/Reshape&^gradients_2/pow_grad/tuple/group_deps*
T0*/
_class%
#!loc:@gradients_2/pow_grad/Reshape*#
_output_shapes
:���������
�
/gradients_2/pow_grad/tuple/control_dependency_1Identitygradients_2/pow_grad/Reshape_1&^gradients_2/pow_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_2/pow_grad/Reshape_1*
_output_shapes
: 
j
gradients_2/pow_1_grad/ShapeShapesub_5^Adam_1*
T0*
out_type0*
_output_shapes
:
l
gradients_2/pow_1_grad/Shape_1Shapepow_1/y^Adam_1*
T0*
out_type0*
_output_shapes
: 
�
,gradients_2/pow_1_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/pow_1_grad/Shapegradients_2/pow_1_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
y
gradients_2/pow_1_grad/mulMulgradients_2/Mean_1_grad/truedivpow_1/y*
T0*#
_output_shapes
:���������
j
gradients_2/pow_1_grad/sub/yConst^Adam_1*
_output_shapes
: *
valueB
 *  �?*
dtype0
i
gradients_2/pow_1_grad/subSubpow_1/ygradients_2/pow_1_grad/sub/y*
_output_shapes
: *
T0
r
gradients_2/pow_1_grad/PowPowsub_5gradients_2/pow_1_grad/sub*
T0*#
_output_shapes
:���������
�
gradients_2/pow_1_grad/mul_1Mulgradients_2/pow_1_grad/mulgradients_2/pow_1_grad/Pow*
T0*#
_output_shapes
:���������
�
gradients_2/pow_1_grad/SumSumgradients_2/pow_1_grad/mul_1,gradients_2/pow_1_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients_2/pow_1_grad/ReshapeReshapegradients_2/pow_1_grad/Sumgradients_2/pow_1_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
n
 gradients_2/pow_1_grad/Greater/yConst^Adam_1*
valueB
 *    *
dtype0*
_output_shapes
: 
�
gradients_2/pow_1_grad/GreaterGreatersub_5 gradients_2/pow_1_grad/Greater/y*
T0*#
_output_shapes
:���������
t
&gradients_2/pow_1_grad/ones_like/ShapeShapesub_5^Adam_1*
T0*
out_type0*
_output_shapes
:
t
&gradients_2/pow_1_grad/ones_like/ConstConst^Adam_1*
valueB
 *  �?*
dtype0*
_output_shapes
: 
�
 gradients_2/pow_1_grad/ones_likeFill&gradients_2/pow_1_grad/ones_like/Shape&gradients_2/pow_1_grad/ones_like/Const*
T0*

index_type0*#
_output_shapes
:���������
�
gradients_2/pow_1_grad/SelectSelectgradients_2/pow_1_grad/Greatersub_5 gradients_2/pow_1_grad/ones_like*
T0*#
_output_shapes
:���������
n
gradients_2/pow_1_grad/LogLoggradients_2/pow_1_grad/Select*
T0*#
_output_shapes
:���������
l
!gradients_2/pow_1_grad/zeros_like	ZerosLikesub_5^Adam_1*#
_output_shapes
:���������*
T0
�
gradients_2/pow_1_grad/Select_1Selectgradients_2/pow_1_grad/Greatergradients_2/pow_1_grad/Log!gradients_2/pow_1_grad/zeros_like*
T0*#
_output_shapes
:���������
y
gradients_2/pow_1_grad/mul_2Mulgradients_2/Mean_1_grad/truedivpow_1*#
_output_shapes
:���������*
T0
�
gradients_2/pow_1_grad/mul_3Mulgradients_2/pow_1_grad/mul_2gradients_2/pow_1_grad/Select_1*#
_output_shapes
:���������*
T0
�
gradients_2/pow_1_grad/Sum_1Sumgradients_2/pow_1_grad/mul_3.gradients_2/pow_1_grad/BroadcastGradientArgs:1*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
 gradients_2/pow_1_grad/Reshape_1Reshapegradients_2/pow_1_grad/Sum_1gradients_2/pow_1_grad/Shape_1*
T0*
Tshape0*
_output_shapes
: 
|
'gradients_2/pow_1_grad/tuple/group_depsNoOp^Adam_1^gradients_2/pow_1_grad/Reshape!^gradients_2/pow_1_grad/Reshape_1
�
/gradients_2/pow_1_grad/tuple/control_dependencyIdentitygradients_2/pow_1_grad/Reshape(^gradients_2/pow_1_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_2/pow_1_grad/Reshape*#
_output_shapes
:���������
�
1gradients_2/pow_1_grad/tuple/control_dependency_1Identity gradients_2/pow_1_grad/Reshape_1(^gradients_2/pow_1_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_2/pow_1_grad/Reshape_1*
_output_shapes
: 
j
gradients_2/sub_4_grad/ShapeShapeadd_4^Adam_1*
T0*
out_type0*
_output_shapes
:
l
gradients_2/sub_4_grad/Shape_1ShapeSum_3^Adam_1*
T0*
out_type0*
_output_shapes
:
�
,gradients_2/sub_4_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_4_grad/Shapegradients_2/sub_4_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_2/sub_4_grad/SumSum-gradients_2/pow_grad/tuple/control_dependency,gradients_2/sub_4_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_2/sub_4_grad/ReshapeReshapegradients_2/sub_4_grad/Sumgradients_2/sub_4_grad/Shape*
T0*
Tshape0*#
_output_shapes
:���������
~
gradients_2/sub_4_grad/NegNeg-gradients_2/pow_grad/tuple/control_dependency*#
_output_shapes
:���������*
T0
�
gradients_2/sub_4_grad/Sum_1Sumgradients_2/sub_4_grad/Neg.gradients_2/sub_4_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
 gradients_2/sub_4_grad/Reshape_1Reshapegradients_2/sub_4_grad/Sum_1gradients_2/sub_4_grad/Shape_1*#
_output_shapes
:���������*
T0*
Tshape0
|
'gradients_2/sub_4_grad/tuple/group_depsNoOp^Adam_1^gradients_2/sub_4_grad/Reshape!^gradients_2/sub_4_grad/Reshape_1
�
/gradients_2/sub_4_grad/tuple/control_dependencyIdentitygradients_2/sub_4_grad/Reshape(^gradients_2/sub_4_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_2/sub_4_grad/Reshape*#
_output_shapes
:���������
�
1gradients_2/sub_4_grad/tuple/control_dependency_1Identity gradients_2/sub_4_grad/Reshape_1(^gradients_2/sub_4_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_2/sub_4_grad/Reshape_1*#
_output_shapes
:���������
j
gradients_2/sub_5_grad/ShapeShapeadd_4^Adam_1*
T0*
out_type0*
_output_shapes
:
l
gradients_2/sub_5_grad/Shape_1ShapeSum_4^Adam_1*
T0*
out_type0*
_output_shapes
:
�
,gradients_2/sub_5_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/sub_5_grad/Shapegradients_2/sub_5_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_2/sub_5_grad/SumSum/gradients_2/pow_1_grad/tuple/control_dependency,gradients_2/sub_5_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_2/sub_5_grad/ReshapeReshapegradients_2/sub_5_grad/Sumgradients_2/sub_5_grad/Shape*#
_output_shapes
:���������*
T0*
Tshape0
�
gradients_2/sub_5_grad/NegNeg/gradients_2/pow_1_grad/tuple/control_dependency*
T0*#
_output_shapes
:���������
�
gradients_2/sub_5_grad/Sum_1Sumgradients_2/sub_5_grad/Neg.gradients_2/sub_5_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
 gradients_2/sub_5_grad/Reshape_1Reshapegradients_2/sub_5_grad/Sum_1gradients_2/sub_5_grad/Shape_1*
T0*
Tshape0*#
_output_shapes
:���������
|
'gradients_2/sub_5_grad/tuple/group_depsNoOp^Adam_1^gradients_2/sub_5_grad/Reshape!^gradients_2/sub_5_grad/Reshape_1
�
/gradients_2/sub_5_grad/tuple/control_dependencyIdentitygradients_2/sub_5_grad/Reshape(^gradients_2/sub_5_grad/tuple/group_deps*#
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients_2/sub_5_grad/Reshape
�
1gradients_2/sub_5_grad/tuple/control_dependency_1Identity gradients_2/sub_5_grad/Reshape_1(^gradients_2/sub_5_grad/tuple/group_deps*3
_class)
'%loc:@gradients_2/sub_5_grad/Reshape_1*#
_output_shapes
:���������*
T0
j
gradients_2/Sum_3_grad/ShapeShapeMul_9^Adam_1*
out_type0*
_output_shapes
:*
T0
�
gradients_2/Sum_3_grad/SizeConst^Adam_1*
value	B :*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients_2/Sum_3_grad/addAddV2Sum_3/reduction_indicesgradients_2/Sum_3_grad/Size*
T0*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
_output_shapes
: 
�
gradients_2/Sum_3_grad/modFloorModgradients_2/Sum_3_grad/addgradients_2/Sum_3_grad/Size*
T0*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
_output_shapes
: 
�
gradients_2/Sum_3_grad/Shape_1Const^Adam_1*
valueB */
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
�
"gradients_2/Sum_3_grad/range/startConst^Adam_1*
_output_shapes
: *
value	B : */
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
dtype0
�
"gradients_2/Sum_3_grad/range/deltaConst^Adam_1*
_output_shapes
: *
value	B :*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
dtype0
�
gradients_2/Sum_3_grad/rangeRange"gradients_2/Sum_3_grad/range/startgradients_2/Sum_3_grad/Size"gradients_2/Sum_3_grad/range/delta*
_output_shapes
:*

Tidx0*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape
�
!gradients_2/Sum_3_grad/Fill/valueConst^Adam_1*
value	B :*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients_2/Sum_3_grad/FillFillgradients_2/Sum_3_grad/Shape_1!gradients_2/Sum_3_grad/Fill/value*
T0*

index_type0*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
_output_shapes
: 
�
$gradients_2/Sum_3_grad/DynamicStitchDynamicStitchgradients_2/Sum_3_grad/rangegradients_2/Sum_3_grad/modgradients_2/Sum_3_grad/Shapegradients_2/Sum_3_grad/Fill*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
N*
_output_shapes
:*
T0
�
 gradients_2/Sum_3_grad/Maximum/yConst^Adam_1*
value	B :*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients_2/Sum_3_grad/MaximumMaximum$gradients_2/Sum_3_grad/DynamicStitch gradients_2/Sum_3_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
_output_shapes
:
�
gradients_2/Sum_3_grad/floordivFloorDivgradients_2/Sum_3_grad/Shapegradients_2/Sum_3_grad/Maximum*
T0*/
_class%
#!loc:@gradients_2/Sum_3_grad/Shape*
_output_shapes
:
�
gradients_2/Sum_3_grad/ReshapeReshape1gradients_2/sub_4_grad/tuple/control_dependency_1$gradients_2/Sum_3_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients_2/Sum_3_grad/TileTilegradients_2/Sum_3_grad/Reshapegradients_2/Sum_3_grad/floordiv*

Tmultiples0*
T0*'
_output_shapes
:���������
k
gradients_2/Sum_4_grad/ShapeShapeMul_10^Adam_1*
T0*
out_type0*
_output_shapes
:
�
gradients_2/Sum_4_grad/SizeConst^Adam_1*
value	B :*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients_2/Sum_4_grad/addAddV2Sum_4/reduction_indicesgradients_2/Sum_4_grad/Size*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape
�
gradients_2/Sum_4_grad/modFloorModgradients_2/Sum_4_grad/addgradients_2/Sum_4_grad/Size*
_output_shapes
: *
T0*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape
�
gradients_2/Sum_4_grad/Shape_1Const^Adam_1*
dtype0*
_output_shapes
: *
valueB */
_class%
#!loc:@gradients_2/Sum_4_grad/Shape
�
"gradients_2/Sum_4_grad/range/startConst^Adam_1*
value	B : */
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
dtype0*
_output_shapes
: 
�
"gradients_2/Sum_4_grad/range/deltaConst^Adam_1*
value	B :*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
dtype0*
_output_shapes
: 
�
gradients_2/Sum_4_grad/rangeRange"gradients_2/Sum_4_grad/range/startgradients_2/Sum_4_grad/Size"gradients_2/Sum_4_grad/range/delta*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
_output_shapes
:*

Tidx0
�
!gradients_2/Sum_4_grad/Fill/valueConst^Adam_1*
_output_shapes
: *
value	B :*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
dtype0
�
gradients_2/Sum_4_grad/FillFillgradients_2/Sum_4_grad/Shape_1!gradients_2/Sum_4_grad/Fill/value*
T0*

index_type0*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
_output_shapes
: 
�
$gradients_2/Sum_4_grad/DynamicStitchDynamicStitchgradients_2/Sum_4_grad/rangegradients_2/Sum_4_grad/modgradients_2/Sum_4_grad/Shapegradients_2/Sum_4_grad/Fill*
T0*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
N*
_output_shapes
:
�
 gradients_2/Sum_4_grad/Maximum/yConst^Adam_1*
_output_shapes
: *
value	B :*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
dtype0
�
gradients_2/Sum_4_grad/MaximumMaximum$gradients_2/Sum_4_grad/DynamicStitch gradients_2/Sum_4_grad/Maximum/y*
T0*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
_output_shapes
:
�
gradients_2/Sum_4_grad/floordivFloorDivgradients_2/Sum_4_grad/Shapegradients_2/Sum_4_grad/Maximum*
T0*/
_class%
#!loc:@gradients_2/Sum_4_grad/Shape*
_output_shapes
:
�
gradients_2/Sum_4_grad/ReshapeReshape1gradients_2/sub_5_grad/tuple/control_dependency_1$gradients_2/Sum_4_grad/DynamicStitch*
T0*
Tshape0*0
_output_shapes
:������������������
�
gradients_2/Sum_4_grad/TileTilegradients_2/Sum_4_grad/Reshapegradients_2/Sum_4_grad/floordiv*'
_output_shapes
:���������*

Tmultiples0*
T0
|
gradients_2/Mul_9_grad/ShapeShapemain/q1/dense_2/BiasAdd^Adam_1*
out_type0*
_output_shapes
:*
T0
t
gradients_2/Mul_9_grad/Shape_1ShapePlaceholder_1^Adam_1*
T0*
out_type0*
_output_shapes
:
�
,gradients_2/Mul_9_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/Mul_9_grad/Shapegradients_2/Mul_9_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0

gradients_2/Mul_9_grad/MulMulgradients_2/Sum_3_grad/TilePlaceholder_1*
T0*'
_output_shapes
:���������
�
gradients_2/Mul_9_grad/SumSumgradients_2/Mul_9_grad/Mul,gradients_2/Mul_9_grad/BroadcastGradientArgs*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
gradients_2/Mul_9_grad/ReshapeReshapegradients_2/Mul_9_grad/Sumgradients_2/Mul_9_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients_2/Mul_9_grad/Mul_1Mulmain/q1/dense_2/BiasAddgradients_2/Sum_3_grad/Tile*'
_output_shapes
:���������*
T0
�
gradients_2/Mul_9_grad/Sum_1Sumgradients_2/Mul_9_grad/Mul_1.gradients_2/Mul_9_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
 gradients_2/Mul_9_grad/Reshape_1Reshapegradients_2/Mul_9_grad/Sum_1gradients_2/Mul_9_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������
|
'gradients_2/Mul_9_grad/tuple/group_depsNoOp^Adam_1^gradients_2/Mul_9_grad/Reshape!^gradients_2/Mul_9_grad/Reshape_1
�
/gradients_2/Mul_9_grad/tuple/control_dependencyIdentitygradients_2/Mul_9_grad/Reshape(^gradients_2/Mul_9_grad/tuple/group_deps*'
_output_shapes
:���������*
T0*1
_class'
%#loc:@gradients_2/Mul_9_grad/Reshape
�
1gradients_2/Mul_9_grad/tuple/control_dependency_1Identity gradients_2/Mul_9_grad/Reshape_1(^gradients_2/Mul_9_grad/tuple/group_deps*
T0*3
_class)
'%loc:@gradients_2/Mul_9_grad/Reshape_1*'
_output_shapes
:���������
}
gradients_2/Mul_10_grad/ShapeShapemain/q2/dense_2/BiasAdd^Adam_1*
_output_shapes
:*
T0*
out_type0
u
gradients_2/Mul_10_grad/Shape_1ShapePlaceholder_1^Adam_1*
T0*
out_type0*
_output_shapes
:
�
-gradients_2/Mul_10_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_2/Mul_10_grad/Shapegradients_2/Mul_10_grad/Shape_1*2
_output_shapes 
:���������:���������*
T0
�
gradients_2/Mul_10_grad/MulMulgradients_2/Sum_4_grad/TilePlaceholder_1*'
_output_shapes
:���������*
T0
�
gradients_2/Mul_10_grad/SumSumgradients_2/Mul_10_grad/Mul-gradients_2/Mul_10_grad/BroadcastGradientArgs*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
gradients_2/Mul_10_grad/ReshapeReshapegradients_2/Mul_10_grad/Sumgradients_2/Mul_10_grad/Shape*
T0*
Tshape0*'
_output_shapes
:���������
�
gradients_2/Mul_10_grad/Mul_1Mulmain/q2/dense_2/BiasAddgradients_2/Sum_4_grad/Tile*
T0*'
_output_shapes
:���������
�
gradients_2/Mul_10_grad/Sum_1Sumgradients_2/Mul_10_grad/Mul_1/gradients_2/Mul_10_grad/BroadcastGradientArgs:1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
:
�
!gradients_2/Mul_10_grad/Reshape_1Reshapegradients_2/Mul_10_grad/Sum_1gradients_2/Mul_10_grad/Shape_1*
T0*
Tshape0*'
_output_shapes
:���������

(gradients_2/Mul_10_grad/tuple/group_depsNoOp^Adam_1 ^gradients_2/Mul_10_grad/Reshape"^gradients_2/Mul_10_grad/Reshape_1
�
0gradients_2/Mul_10_grad/tuple/control_dependencyIdentitygradients_2/Mul_10_grad/Reshape)^gradients_2/Mul_10_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_2/Mul_10_grad/Reshape*'
_output_shapes
:���������
�
2gradients_2/Mul_10_grad/tuple/control_dependency_1Identity!gradients_2/Mul_10_grad/Reshape_1)^gradients_2/Mul_10_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_2/Mul_10_grad/Reshape_1*'
_output_shapes
:���������
�
4gradients_2/main/q1/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad/gradients_2/Mul_9_grad/tuple/control_dependency*
T0*
data_formatNHWC*
_output_shapes
:
�
9gradients_2/main/q1/dense_2/BiasAdd_grad/tuple/group_depsNoOp^Adam_10^gradients_2/Mul_9_grad/tuple/control_dependency5^gradients_2/main/q1/dense_2/BiasAdd_grad/BiasAddGrad
�
Agradients_2/main/q1/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity/gradients_2/Mul_9_grad/tuple/control_dependency:^gradients_2/main/q1/dense_2/BiasAdd_grad/tuple/group_deps*
T0*1
_class'
%#loc:@gradients_2/Mul_9_grad/Reshape*'
_output_shapes
:���������
�
Cgradients_2/main/q1/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_2/main/q1/dense_2/BiasAdd_grad/BiasAddGrad:^gradients_2/main/q1/dense_2/BiasAdd_grad/tuple/group_deps*G
_class=
;9loc:@gradients_2/main/q1/dense_2/BiasAdd_grad/BiasAddGrad*
_output_shapes
:*
T0
�
4gradients_2/main/q2/dense_2/BiasAdd_grad/BiasAddGradBiasAddGrad0gradients_2/Mul_10_grad/tuple/control_dependency*
data_formatNHWC*
_output_shapes
:*
T0
�
9gradients_2/main/q2/dense_2/BiasAdd_grad/tuple/group_depsNoOp^Adam_11^gradients_2/Mul_10_grad/tuple/control_dependency5^gradients_2/main/q2/dense_2/BiasAdd_grad/BiasAddGrad
�
Agradients_2/main/q2/dense_2/BiasAdd_grad/tuple/control_dependencyIdentity0gradients_2/Mul_10_grad/tuple/control_dependency:^gradients_2/main/q2/dense_2/BiasAdd_grad/tuple/group_deps*2
_class(
&$loc:@gradients_2/Mul_10_grad/Reshape*'
_output_shapes
:���������*
T0
�
Cgradients_2/main/q2/dense_2/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_2/main/q2/dense_2/BiasAdd_grad/BiasAddGrad:^gradients_2/main/q2/dense_2/BiasAdd_grad/tuple/group_deps*
_output_shapes
:*
T0*G
_class=
;9loc:@gradients_2/main/q2/dense_2/BiasAdd_grad/BiasAddGrad
�
.gradients_2/main/q1/dense_2/MatMul_grad/MatMulMatMulAgradients_2/main/q1/dense_2/BiasAdd_grad/tuple/control_dependencymain/q1/dense_2/kernel/read*
transpose_b(*
T0*'
_output_shapes
:���������@*
transpose_a( 
�
0gradients_2/main/q1/dense_2/MatMul_grad/MatMul_1MatMulmain/q1/dense_1/ReluAgradients_2/main/q1/dense_2/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
8gradients_2/main/q1/dense_2/MatMul_grad/tuple/group_depsNoOp^Adam_1/^gradients_2/main/q1/dense_2/MatMul_grad/MatMul1^gradients_2/main/q1/dense_2/MatMul_grad/MatMul_1
�
@gradients_2/main/q1/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients_2/main/q1/dense_2/MatMul_grad/MatMul9^gradients_2/main/q1/dense_2/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/main/q1/dense_2/MatMul_grad/MatMul*'
_output_shapes
:���������@
�
Bgradients_2/main/q1/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients_2/main/q1/dense_2/MatMul_grad/MatMul_19^gradients_2/main/q1/dense_2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/main/q1/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
�
.gradients_2/main/q2/dense_2/MatMul_grad/MatMulMatMulAgradients_2/main/q2/dense_2/BiasAdd_grad/tuple/control_dependencymain/q2/dense_2/kernel/read*
transpose_b(*
T0*'
_output_shapes
:���������@*
transpose_a( 
�
0gradients_2/main/q2/dense_2/MatMul_grad/MatMul_1MatMulmain/q2/dense_1/ReluAgradients_2/main/q2/dense_2/BiasAdd_grad/tuple/control_dependency*
transpose_b( *
T0*
_output_shapes

:@*
transpose_a(
�
8gradients_2/main/q2/dense_2/MatMul_grad/tuple/group_depsNoOp^Adam_1/^gradients_2/main/q2/dense_2/MatMul_grad/MatMul1^gradients_2/main/q2/dense_2/MatMul_grad/MatMul_1
�
@gradients_2/main/q2/dense_2/MatMul_grad/tuple/control_dependencyIdentity.gradients_2/main/q2/dense_2/MatMul_grad/MatMul9^gradients_2/main/q2/dense_2/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients_2/main/q2/dense_2/MatMul_grad/MatMul*'
_output_shapes
:���������@*
T0
�
Bgradients_2/main/q2/dense_2/MatMul_grad/tuple/control_dependency_1Identity0gradients_2/main/q2/dense_2/MatMul_grad/MatMul_19^gradients_2/main/q2/dense_2/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/main/q2/dense_2/MatMul_grad/MatMul_1*
_output_shapes

:@
�
.gradients_2/main/q1/dense_1/Relu_grad/ReluGradReluGrad@gradients_2/main/q1/dense_2/MatMul_grad/tuple/control_dependencymain/q1/dense_1/Relu*
T0*'
_output_shapes
:���������@
�
.gradients_2/main/q2/dense_1/Relu_grad/ReluGradReluGrad@gradients_2/main/q2/dense_2/MatMul_grad/tuple/control_dependencymain/q2/dense_1/Relu*
T0*'
_output_shapes
:���������@
�
4gradients_2/main/q1/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_2/main/q1/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
9gradients_2/main/q1/dense_1/BiasAdd_grad/tuple/group_depsNoOp^Adam_15^gradients_2/main/q1/dense_1/BiasAdd_grad/BiasAddGrad/^gradients_2/main/q1/dense_1/Relu_grad/ReluGrad
�
Agradients_2/main/q1/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients_2/main/q1/dense_1/Relu_grad/ReluGrad:^gradients_2/main/q1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/main/q1/dense_1/Relu_grad/ReluGrad*'
_output_shapes
:���������@
�
Cgradients_2/main/q1/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_2/main/q1/dense_1/BiasAdd_grad/BiasAddGrad:^gradients_2/main/q1/dense_1/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_2/main/q1/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
4gradients_2/main/q2/dense_1/BiasAdd_grad/BiasAddGradBiasAddGrad.gradients_2/main/q2/dense_1/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
9gradients_2/main/q2/dense_1/BiasAdd_grad/tuple/group_depsNoOp^Adam_15^gradients_2/main/q2/dense_1/BiasAdd_grad/BiasAddGrad/^gradients_2/main/q2/dense_1/Relu_grad/ReluGrad
�
Agradients_2/main/q2/dense_1/BiasAdd_grad/tuple/control_dependencyIdentity.gradients_2/main/q2/dense_1/Relu_grad/ReluGrad:^gradients_2/main/q2/dense_1/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������@*
T0*A
_class7
53loc:@gradients_2/main/q2/dense_1/Relu_grad/ReluGrad
�
Cgradients_2/main/q2/dense_1/BiasAdd_grad/tuple/control_dependency_1Identity4gradients_2/main/q2/dense_1/BiasAdd_grad/BiasAddGrad:^gradients_2/main/q2/dense_1/BiasAdd_grad/tuple/group_deps*
T0*G
_class=
;9loc:@gradients_2/main/q2/dense_1/BiasAdd_grad/BiasAddGrad*
_output_shapes
:@
�
.gradients_2/main/q1/dense_1/MatMul_grad/MatMulMatMulAgradients_2/main/q1/dense_1/BiasAdd_grad/tuple/control_dependencymain/q1/dense_1/kernel/read*
transpose_b(*
T0*'
_output_shapes
:���������@*
transpose_a( 
�
0gradients_2/main/q1/dense_1/MatMul_grad/MatMul_1MatMulmain/q1/dense/ReluAgradients_2/main/q1/dense_1/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@@*
transpose_a(*
transpose_b( *
T0
�
8gradients_2/main/q1/dense_1/MatMul_grad/tuple/group_depsNoOp^Adam_1/^gradients_2/main/q1/dense_1/MatMul_grad/MatMul1^gradients_2/main/q1/dense_1/MatMul_grad/MatMul_1
�
@gradients_2/main/q1/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_2/main/q1/dense_1/MatMul_grad/MatMul9^gradients_2/main/q1/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/main/q1/dense_1/MatMul_grad/MatMul*'
_output_shapes
:���������@
�
Bgradients_2/main/q1/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_2/main/q1/dense_1/MatMul_grad/MatMul_19^gradients_2/main/q1/dense_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/main/q1/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
.gradients_2/main/q2/dense_1/MatMul_grad/MatMulMatMulAgradients_2/main/q2/dense_1/BiasAdd_grad/tuple/control_dependencymain/q2/dense_1/kernel/read*
T0*'
_output_shapes
:���������@*
transpose_a( *
transpose_b(
�
0gradients_2/main/q2/dense_1/MatMul_grad/MatMul_1MatMulmain/q2/dense/ReluAgradients_2/main/q2/dense_1/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@@*
transpose_a(*
transpose_b( 
�
8gradients_2/main/q2/dense_1/MatMul_grad/tuple/group_depsNoOp^Adam_1/^gradients_2/main/q2/dense_1/MatMul_grad/MatMul1^gradients_2/main/q2/dense_1/MatMul_grad/MatMul_1
�
@gradients_2/main/q2/dense_1/MatMul_grad/tuple/control_dependencyIdentity.gradients_2/main/q2/dense_1/MatMul_grad/MatMul9^gradients_2/main/q2/dense_1/MatMul_grad/tuple/group_deps*
T0*A
_class7
53loc:@gradients_2/main/q2/dense_1/MatMul_grad/MatMul*'
_output_shapes
:���������@
�
Bgradients_2/main/q2/dense_1/MatMul_grad/tuple/control_dependency_1Identity0gradients_2/main/q2/dense_1/MatMul_grad/MatMul_19^gradients_2/main/q2/dense_1/MatMul_grad/tuple/group_deps*
T0*C
_class9
75loc:@gradients_2/main/q2/dense_1/MatMul_grad/MatMul_1*
_output_shapes

:@@
�
,gradients_2/main/q1/dense/Relu_grad/ReluGradReluGrad@gradients_2/main/q1/dense_1/MatMul_grad/tuple/control_dependencymain/q1/dense/Relu*'
_output_shapes
:���������@*
T0
�
,gradients_2/main/q2/dense/Relu_grad/ReluGradReluGrad@gradients_2/main/q2/dense_1/MatMul_grad/tuple/control_dependencymain/q2/dense/Relu*
T0*'
_output_shapes
:���������@
�
2gradients_2/main/q1/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_2/main/q1/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
7gradients_2/main/q1/dense/BiasAdd_grad/tuple/group_depsNoOp^Adam_13^gradients_2/main/q1/dense/BiasAdd_grad/BiasAddGrad-^gradients_2/main/q1/dense/Relu_grad/ReluGrad
�
?gradients_2/main/q1/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_2/main/q1/dense/Relu_grad/ReluGrad8^gradients_2/main/q1/dense/BiasAdd_grad/tuple/group_deps*'
_output_shapes
:���������@*
T0*?
_class5
31loc:@gradients_2/main/q1/dense/Relu_grad/ReluGrad
�
Agradients_2/main/q1/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_2/main/q1/dense/BiasAdd_grad/BiasAddGrad8^gradients_2/main/q1/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*E
_class;
97loc:@gradients_2/main/q1/dense/BiasAdd_grad/BiasAddGrad
�
2gradients_2/main/q2/dense/BiasAdd_grad/BiasAddGradBiasAddGrad,gradients_2/main/q2/dense/Relu_grad/ReluGrad*
T0*
data_formatNHWC*
_output_shapes
:@
�
7gradients_2/main/q2/dense/BiasAdd_grad/tuple/group_depsNoOp^Adam_13^gradients_2/main/q2/dense/BiasAdd_grad/BiasAddGrad-^gradients_2/main/q2/dense/Relu_grad/ReluGrad
�
?gradients_2/main/q2/dense/BiasAdd_grad/tuple/control_dependencyIdentity,gradients_2/main/q2/dense/Relu_grad/ReluGrad8^gradients_2/main/q2/dense/BiasAdd_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_2/main/q2/dense/Relu_grad/ReluGrad*'
_output_shapes
:���������@
�
Agradients_2/main/q2/dense/BiasAdd_grad/tuple/control_dependency_1Identity2gradients_2/main/q2/dense/BiasAdd_grad/BiasAddGrad8^gradients_2/main/q2/dense/BiasAdd_grad/tuple/group_deps*
_output_shapes
:@*
T0*E
_class;
97loc:@gradients_2/main/q2/dense/BiasAdd_grad/BiasAddGrad
�
,gradients_2/main/q1/dense/MatMul_grad/MatMulMatMul?gradients_2/main/q1/dense/BiasAdd_grad/tuple/control_dependencymain/q1/dense/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
.gradients_2/main/q1/dense/MatMul_grad/MatMul_1MatMulPlaceholder?gradients_2/main/q1/dense/BiasAdd_grad/tuple/control_dependency*
T0*
_output_shapes

:@*
transpose_a(*
transpose_b( 
�
6gradients_2/main/q1/dense/MatMul_grad/tuple/group_depsNoOp^Adam_1-^gradients_2/main/q1/dense/MatMul_grad/MatMul/^gradients_2/main/q1/dense/MatMul_grad/MatMul_1
�
>gradients_2/main/q1/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_2/main/q1/dense/MatMul_grad/MatMul7^gradients_2/main/q1/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_2/main/q1/dense/MatMul_grad/MatMul*'
_output_shapes
:���������
�
@gradients_2/main/q1/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_2/main/q1/dense/MatMul_grad/MatMul_17^gradients_2/main/q1/dense/MatMul_grad/tuple/group_deps*
_output_shapes

:@*
T0*A
_class7
53loc:@gradients_2/main/q1/dense/MatMul_grad/MatMul_1
�
,gradients_2/main/q2/dense/MatMul_grad/MatMulMatMul?gradients_2/main/q2/dense/BiasAdd_grad/tuple/control_dependencymain/q2/dense/kernel/read*'
_output_shapes
:���������*
transpose_a( *
transpose_b(*
T0
�
.gradients_2/main/q2/dense/MatMul_grad/MatMul_1MatMulPlaceholder?gradients_2/main/q2/dense/BiasAdd_grad/tuple/control_dependency*
_output_shapes

:@*
transpose_a(*
transpose_b( *
T0
�
6gradients_2/main/q2/dense/MatMul_grad/tuple/group_depsNoOp^Adam_1-^gradients_2/main/q2/dense/MatMul_grad/MatMul/^gradients_2/main/q2/dense/MatMul_grad/MatMul_1
�
>gradients_2/main/q2/dense/MatMul_grad/tuple/control_dependencyIdentity,gradients_2/main/q2/dense/MatMul_grad/MatMul7^gradients_2/main/q2/dense/MatMul_grad/tuple/group_deps*
T0*?
_class5
31loc:@gradients_2/main/q2/dense/MatMul_grad/MatMul*'
_output_shapes
:���������
�
@gradients_2/main/q2/dense/MatMul_grad/tuple/control_dependency_1Identity.gradients_2/main/q2/dense/MatMul_grad/MatMul_17^gradients_2/main/q2/dense/MatMul_grad/tuple/group_deps*A
_class7
53loc:@gradients_2/main/q2/dense/MatMul_grad/MatMul_1*
_output_shapes

:@*
T0
�
beta1_power_2/initial_valueConst*
valueB
 *fff?*%
_class
loc:@main/q1/dense/bias*
dtype0*
_output_shapes
: 
�
beta1_power_2
VariableV2*
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@main/q1/dense/bias*
	container *
shape: 
�
beta1_power_2/AssignAssignbeta1_power_2beta1_power_2/initial_value*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: 
u
beta1_power_2/readIdentitybeta1_power_2*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: *
T0
�
beta2_power_2/initial_valueConst*
_output_shapes
: *
valueB
 *w�?*%
_class
loc:@main/q1/dense/bias*
dtype0
�
beta2_power_2
VariableV2*
dtype0*
_output_shapes
: *
shared_name *%
_class
loc:@main/q1/dense/bias*
	container *
shape: 
�
beta2_power_2/AssignAssignbeta2_power_2beta2_power_2/initial_value*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: 
u
beta2_power_2/readIdentitybeta2_power_2*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
�
+main/q1/dense/kernel/Adam/Initializer/zerosConst*'
_class
loc:@main/q1/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/q1/dense/kernel/Adam
VariableV2*
shared_name *'
_class
loc:@main/q1/dense/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@
�
 main/q1/dense/kernel/Adam/AssignAssignmain/q1/dense/kernel/Adam+main/q1/dense/kernel/Adam/Initializer/zeros*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
main/q1/dense/kernel/Adam/readIdentitymain/q1/dense/kernel/Adam*
_output_shapes

:@*
T0*'
_class
loc:@main/q1/dense/kernel
�
-main/q1/dense/kernel/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q1/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/q1/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *'
_class
loc:@main/q1/dense/kernel*
	container *
shape
:@
�
"main/q1/dense/kernel/Adam_1/AssignAssignmain/q1/dense/kernel/Adam_1-main/q1/dense/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
 main/q1/dense/kernel/Adam_1/readIdentitymain/q1/dense/kernel/Adam_1*
T0*'
_class
loc:@main/q1/dense/kernel*
_output_shapes

:@
�
)main/q1/dense/bias/Adam/Initializer/zerosConst*%
_class
loc:@main/q1/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/q1/dense/bias/Adam
VariableV2*%
_class
loc:@main/q1/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
main/q1/dense/bias/Adam/AssignAssignmain/q1/dense/bias/Adam)main/q1/dense/bias/Adam/Initializer/zeros*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
main/q1/dense/bias/Adam/readIdentitymain/q1/dense/bias/Adam*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
:@
�
+main/q1/dense/bias/Adam_1/Initializer/zerosConst*%
_class
loc:@main/q1/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/q1/dense/bias/Adam_1
VariableV2*%
_class
loc:@main/q1/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
 main/q1/dense/bias/Adam_1/AssignAssignmain/q1/dense/bias/Adam_1+main/q1/dense/bias/Adam_1/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(
�
main/q1/dense/bias/Adam_1/readIdentitymain/q1/dense/bias/Adam_1*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
:@
�
=main/q1/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/q1/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
3main/q1/dense_1/kernel/Adam/Initializer/zeros/ConstConst*
_output_shapes
: *)
_class
loc:@main/q1/dense_1/kernel*
valueB
 *    *
dtype0
�
-main/q1/dense_1/kernel/Adam/Initializer/zerosFill=main/q1/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/q1/dense_1/kernel/Adam/Initializer/zeros/Const*
T0*)
_class
loc:@main/q1/dense_1/kernel*

index_type0*
_output_shapes

:@@
�
main/q1/dense_1/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:@@*
shared_name *)
_class
loc:@main/q1/dense_1/kernel*
	container *
shape
:@@
�
"main/q1/dense_1/kernel/Adam/AssignAssignmain/q1/dense_1/kernel/Adam-main/q1/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
 main/q1/dense_1/kernel/Adam/readIdentitymain/q1/dense_1/kernel/Adam*
_output_shapes

:@@*
T0*)
_class
loc:@main/q1/dense_1/kernel
�
?main/q1/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/q1/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
5main/q1/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@main/q1/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/main/q1/dense_1/kernel/Adam_1/Initializer/zerosFill?main/q1/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/q1/dense_1/kernel/Adam_1/Initializer/zeros/Const*
T0*)
_class
loc:@main/q1/dense_1/kernel*

index_type0*
_output_shapes

:@@
�
main/q1/dense_1/kernel/Adam_1
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *)
_class
loc:@main/q1/dense_1/kernel
�
$main/q1/dense_1/kernel/Adam_1/AssignAssignmain/q1/dense_1/kernel/Adam_1/main/q1/dense_1/kernel/Adam_1/Initializer/zeros*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
"main/q1/dense_1/kernel/Adam_1/readIdentitymain/q1/dense_1/kernel/Adam_1*
T0*)
_class
loc:@main/q1/dense_1/kernel*
_output_shapes

:@@
�
+main/q1/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/q1/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/q1/dense_1/bias/Adam
VariableV2*
shared_name *'
_class
loc:@main/q1/dense_1/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@
�
 main/q1/dense_1/bias/Adam/AssignAssignmain/q1/dense_1/bias/Adam+main/q1/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
main/q1/dense_1/bias/Adam/readIdentitymain/q1/dense_1/bias/Adam*'
_class
loc:@main/q1/dense_1/bias*
_output_shapes
:@*
T0
�
-main/q1/dense_1/bias/Adam_1/Initializer/zerosConst*
dtype0*
_output_shapes
:@*'
_class
loc:@main/q1/dense_1/bias*
valueB@*    
�
main/q1/dense_1/bias/Adam_1
VariableV2*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@main/q1/dense_1/bias*
	container *
shape:@
�
"main/q1/dense_1/bias/Adam_1/AssignAssignmain/q1/dense_1/bias/Adam_1-main/q1/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
 main/q1/dense_1/bias/Adam_1/readIdentitymain/q1/dense_1/bias/Adam_1*
T0*'
_class
loc:@main/q1/dense_1/bias*
_output_shapes
:@
�
-main/q1/dense_2/kernel/Adam/Initializer/zerosConst*)
_class
loc:@main/q1/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/q1/dense_2/kernel/Adam
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *)
_class
loc:@main/q1/dense_2/kernel
�
"main/q1/dense_2/kernel/Adam/AssignAssignmain/q1/dense_2/kernel/Adam-main/q1/dense_2/kernel/Adam/Initializer/zeros*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
 main/q1/dense_2/kernel/Adam/readIdentitymain/q1/dense_2/kernel/Adam*
_output_shapes

:@*
T0*)
_class
loc:@main/q1/dense_2/kernel
�
/main/q1/dense_2/kernel/Adam_1/Initializer/zerosConst*)
_class
loc:@main/q1/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/q1/dense_2/kernel/Adam_1
VariableV2*
_output_shapes

:@*
shared_name *)
_class
loc:@main/q1/dense_2/kernel*
	container *
shape
:@*
dtype0
�
$main/q1/dense_2/kernel/Adam_1/AssignAssignmain/q1/dense_2/kernel/Adam_1/main/q1/dense_2/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
"main/q1/dense_2/kernel/Adam_1/readIdentitymain/q1/dense_2/kernel/Adam_1*
T0*)
_class
loc:@main/q1/dense_2/kernel*
_output_shapes

:@
�
+main/q1/dense_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/q1/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
�
main/q1/dense_2/bias/Adam
VariableV2*
shared_name *'
_class
loc:@main/q1/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
 main/q1/dense_2/bias/Adam/AssignAssignmain/q1/dense_2/bias/Adam+main/q1/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
�
main/q1/dense_2/bias/Adam/readIdentitymain/q1/dense_2/bias/Adam*
T0*'
_class
loc:@main/q1/dense_2/bias*
_output_shapes
:
�
-main/q1/dense_2/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q1/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
�
main/q1/dense_2/bias/Adam_1
VariableV2*
shared_name *'
_class
loc:@main/q1/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
"main/q1/dense_2/bias/Adam_1/AssignAssignmain/q1/dense_2/bias/Adam_1-main/q1/dense_2/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
�
 main/q1/dense_2/bias/Adam_1/readIdentitymain/q1/dense_2/bias/Adam_1*
T0*'
_class
loc:@main/q1/dense_2/bias*
_output_shapes
:
�
+main/q2/dense/kernel/Adam/Initializer/zerosConst*'
_class
loc:@main/q2/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/q2/dense/kernel/Adam
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *'
_class
loc:@main/q2/dense/kernel*
	container *
shape
:@
�
 main/q2/dense/kernel/Adam/AssignAssignmain/q2/dense/kernel/Adam+main/q2/dense/kernel/Adam/Initializer/zeros*
_output_shapes

:@*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(
�
main/q2/dense/kernel/Adam/readIdentitymain/q2/dense/kernel/Adam*
T0*'
_class
loc:@main/q2/dense/kernel*
_output_shapes

:@
�
-main/q2/dense/kernel/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q2/dense/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/q2/dense/kernel/Adam_1
VariableV2*
dtype0*
_output_shapes

:@*
shared_name *'
_class
loc:@main/q2/dense/kernel*
	container *
shape
:@
�
"main/q2/dense/kernel/Adam_1/AssignAssignmain/q2/dense/kernel/Adam_1-main/q2/dense/kernel/Adam_1/Initializer/zeros*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
 main/q2/dense/kernel/Adam_1/readIdentitymain/q2/dense/kernel/Adam_1*
_output_shapes

:@*
T0*'
_class
loc:@main/q2/dense/kernel
�
)main/q2/dense/bias/Adam/Initializer/zerosConst*%
_class
loc:@main/q2/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/q2/dense/bias/Adam
VariableV2*%
_class
loc:@main/q2/dense/bias*
	container *
shape:@*
dtype0*
_output_shapes
:@*
shared_name 
�
main/q2/dense/bias/Adam/AssignAssignmain/q2/dense/bias/Adam)main/q2/dense/bias/Adam/Initializer/zeros*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(
�
main/q2/dense/bias/Adam/readIdentitymain/q2/dense/bias/Adam*
T0*%
_class
loc:@main/q2/dense/bias*
_output_shapes
:@
�
+main/q2/dense/bias/Adam_1/Initializer/zerosConst*%
_class
loc:@main/q2/dense/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/q2/dense/bias/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *%
_class
loc:@main/q2/dense/bias*
	container 
�
 main/q2/dense/bias/Adam_1/AssignAssignmain/q2/dense/bias/Adam_1+main/q2/dense/bias/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias
�
main/q2/dense/bias/Adam_1/readIdentitymain/q2/dense/bias/Adam_1*
T0*%
_class
loc:@main/q2/dense/bias*
_output_shapes
:@
�
=main/q2/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/q2/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
3main/q2/dense_1/kernel/Adam/Initializer/zeros/ConstConst*)
_class
loc:@main/q2/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
-main/q2/dense_1/kernel/Adam/Initializer/zerosFill=main/q2/dense_1/kernel/Adam/Initializer/zeros/shape_as_tensor3main/q2/dense_1/kernel/Adam/Initializer/zeros/Const*)
_class
loc:@main/q2/dense_1/kernel*

index_type0*
_output_shapes

:@@*
T0
�
main/q2/dense_1/kernel/Adam
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *)
_class
loc:@main/q2/dense_1/kernel
�
"main/q2/dense_1/kernel/Adam/AssignAssignmain/q2/dense_1/kernel/Adam-main/q2/dense_1/kernel/Adam/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
 main/q2/dense_1/kernel/Adam/readIdentitymain/q2/dense_1/kernel/Adam*
T0*)
_class
loc:@main/q2/dense_1/kernel*
_output_shapes

:@@
�
?main/q2/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensorConst*)
_class
loc:@main/q2/dense_1/kernel*
valueB"@   @   *
dtype0*
_output_shapes
:
�
5main/q2/dense_1/kernel/Adam_1/Initializer/zeros/ConstConst*)
_class
loc:@main/q2/dense_1/kernel*
valueB
 *    *
dtype0*
_output_shapes
: 
�
/main/q2/dense_1/kernel/Adam_1/Initializer/zerosFill?main/q2/dense_1/kernel/Adam_1/Initializer/zeros/shape_as_tensor5main/q2/dense_1/kernel/Adam_1/Initializer/zeros/Const*)
_class
loc:@main/q2/dense_1/kernel*

index_type0*
_output_shapes

:@@*
T0
�
main/q2/dense_1/kernel/Adam_1
VariableV2*
	container *
shape
:@@*
dtype0*
_output_shapes

:@@*
shared_name *)
_class
loc:@main/q2/dense_1/kernel
�
$main/q2/dense_1/kernel/Adam_1/AssignAssignmain/q2/dense_1/kernel/Adam_1/main/q2/dense_1/kernel/Adam_1/Initializer/zeros*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
"main/q2/dense_1/kernel/Adam_1/readIdentitymain/q2/dense_1/kernel/Adam_1*
T0*)
_class
loc:@main/q2/dense_1/kernel*
_output_shapes

:@@
�
+main/q2/dense_1/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/q2/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/q2/dense_1/bias/Adam
VariableV2*
_output_shapes
:@*
shared_name *'
_class
loc:@main/q2/dense_1/bias*
	container *
shape:@*
dtype0
�
 main/q2/dense_1/bias/Adam/AssignAssignmain/q2/dense_1/bias/Adam+main/q2/dense_1/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
main/q2/dense_1/bias/Adam/readIdentitymain/q2/dense_1/bias/Adam*
T0*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes
:@
�
-main/q2/dense_1/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q2/dense_1/bias*
valueB@*    *
dtype0*
_output_shapes
:@
�
main/q2/dense_1/bias/Adam_1
VariableV2*
shape:@*
dtype0*
_output_shapes
:@*
shared_name *'
_class
loc:@main/q2/dense_1/bias*
	container 
�
"main/q2/dense_1/bias/Adam_1/AssignAssignmain/q2/dense_1/bias/Adam_1-main/q2/dense_1/bias/Adam_1/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
 main/q2/dense_1/bias/Adam_1/readIdentitymain/q2/dense_1/bias/Adam_1*'
_class
loc:@main/q2/dense_1/bias*
_output_shapes
:@*
T0
�
-main/q2/dense_2/kernel/Adam/Initializer/zerosConst*)
_class
loc:@main/q2/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/q2/dense_2/kernel/Adam
VariableV2*)
_class
loc:@main/q2/dense_2/kernel*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name 
�
"main/q2/dense_2/kernel/Adam/AssignAssignmain/q2/dense_2/kernel/Adam-main/q2/dense_2/kernel/Adam/Initializer/zeros*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
 main/q2/dense_2/kernel/Adam/readIdentitymain/q2/dense_2/kernel/Adam*
_output_shapes

:@*
T0*)
_class
loc:@main/q2/dense_2/kernel
�
/main/q2/dense_2/kernel/Adam_1/Initializer/zerosConst*)
_class
loc:@main/q2/dense_2/kernel*
valueB@*    *
dtype0*
_output_shapes

:@
�
main/q2/dense_2/kernel/Adam_1
VariableV2*
	container *
shape
:@*
dtype0*
_output_shapes

:@*
shared_name *)
_class
loc:@main/q2/dense_2/kernel
�
$main/q2/dense_2/kernel/Adam_1/AssignAssignmain/q2/dense_2/kernel/Adam_1/main/q2/dense_2/kernel/Adam_1/Initializer/zeros*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel
�
"main/q2/dense_2/kernel/Adam_1/readIdentitymain/q2/dense_2/kernel/Adam_1*
T0*)
_class
loc:@main/q2/dense_2/kernel*
_output_shapes

:@
�
+main/q2/dense_2/bias/Adam/Initializer/zerosConst*'
_class
loc:@main/q2/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
�
main/q2/dense_2/bias/Adam
VariableV2*
shared_name *'
_class
loc:@main/q2/dense_2/bias*
	container *
shape:*
dtype0*
_output_shapes
:
�
 main/q2/dense_2/bias/Adam/AssignAssignmain/q2/dense_2/bias/Adam+main/q2/dense_2/bias/Adam/Initializer/zeros*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
main/q2/dense_2/bias/Adam/readIdentitymain/q2/dense_2/bias/Adam*
_output_shapes
:*
T0*'
_class
loc:@main/q2/dense_2/bias
�
-main/q2/dense_2/bias/Adam_1/Initializer/zerosConst*'
_class
loc:@main/q2/dense_2/bias*
valueB*    *
dtype0*
_output_shapes
:
�
main/q2/dense_2/bias/Adam_1
VariableV2*
_output_shapes
:*
shared_name *'
_class
loc:@main/q2/dense_2/bias*
	container *
shape:*
dtype0
�
"main/q2/dense_2/bias/Adam_1/AssignAssignmain/q2/dense_2/bias/Adam_1-main/q2/dense_2/bias/Adam_1/Initializer/zeros*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
 main/q2/dense_2/bias/Adam_1/readIdentitymain/q2/dense_2/bias/Adam_1*
T0*'
_class
loc:@main/q2/dense_2/bias*
_output_shapes
:
b
Adam_2/learning_rateConst^Adam_1*
_output_shapes
: *
valueB
 *RI�9*
dtype0
Z
Adam_2/beta1Const^Adam_1*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Z
Adam_2/beta2Const^Adam_1*
dtype0*
_output_shapes
: *
valueB
 *w�?
\
Adam_2/epsilonConst^Adam_1*
valueB
 *��8*
dtype0*
_output_shapes
: 
�
,Adam_2/update_main/q1/dense/kernel/ApplyAdam	ApplyAdammain/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon@gradients_2/main/q1/dense/MatMul_grad/tuple/control_dependency_1*
T0*'
_class
loc:@main/q1/dense/kernel*
use_nesterov( *
_output_shapes

:@*
use_locking( 
�
*Adam_2/update_main/q1/dense/bias/ApplyAdam	ApplyAdammain/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonAgradients_2/main/q1/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@main/q1/dense/bias*
use_nesterov( *
_output_shapes
:@
�
.Adam_2/update_main/q1/dense_1/kernel/ApplyAdam	ApplyAdammain/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonBgradients_2/main/q1/dense_1/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@main/q1/dense_1/kernel*
use_nesterov( *
_output_shapes

:@@
�
,Adam_2/update_main/q1/dense_1/bias/ApplyAdam	ApplyAdammain/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonCgradients_2/main/q1/dense_1/BiasAdd_grad/tuple/control_dependency_1*'
_class
loc:@main/q1/dense_1/bias*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0
�
.Adam_2/update_main/q1/dense_2/kernel/ApplyAdam	ApplyAdammain/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonBgradients_2/main/q1/dense_2/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*)
_class
loc:@main/q1/dense_2/kernel*
use_nesterov( *
_output_shapes

:@
�
,Adam_2/update_main/q1/dense_2/bias/ApplyAdam	ApplyAdammain/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonCgradients_2/main/q1/dense_2/BiasAdd_grad/tuple/control_dependency_1*'
_class
loc:@main/q1/dense_2/bias*
use_nesterov( *
_output_shapes
:*
use_locking( *
T0
�
,Adam_2/update_main/q2/dense/kernel/ApplyAdam	ApplyAdammain/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilon@gradients_2/main/q2/dense/MatMul_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/q2/dense/kernel*
use_nesterov( *
_output_shapes

:@
�
*Adam_2/update_main/q2/dense/bias/ApplyAdam	ApplyAdammain/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonAgradients_2/main/q2/dense/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*%
_class
loc:@main/q2/dense/bias*
use_nesterov( *
_output_shapes
:@
�
.Adam_2/update_main/q2/dense_1/kernel/ApplyAdam	ApplyAdammain/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonBgradients_2/main/q2/dense_1/MatMul_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes

:@@*
use_locking( *
T0*)
_class
loc:@main/q2/dense_1/kernel
�
,Adam_2/update_main/q2/dense_1/bias/ApplyAdam	ApplyAdammain/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonCgradients_2/main/q2/dense_1/BiasAdd_grad/tuple/control_dependency_1*
use_nesterov( *
_output_shapes
:@*
use_locking( *
T0*'
_class
loc:@main/q2/dense_1/bias
�
.Adam_2/update_main/q2/dense_2/kernel/ApplyAdam	ApplyAdammain/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonBgradients_2/main/q2/dense_2/MatMul_grad/tuple/control_dependency_1*
_output_shapes

:@*
use_locking( *
T0*)
_class
loc:@main/q2/dense_2/kernel*
use_nesterov( 
�
,Adam_2/update_main/q2/dense_2/bias/ApplyAdam	ApplyAdammain/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1beta1_power_2/readbeta2_power_2/readAdam_2/learning_rateAdam_2/beta1Adam_2/beta2Adam_2/epsilonCgradients_2/main/q2/dense_2/BiasAdd_grad/tuple/control_dependency_1*
use_locking( *
T0*'
_class
loc:@main/q2/dense_2/bias*
use_nesterov( *
_output_shapes
:
�

Adam_2/mulMulbeta1_power_2/readAdam_2/beta1+^Adam_2/update_main/q1/dense/bias/ApplyAdam-^Adam_2/update_main/q1/dense/kernel/ApplyAdam-^Adam_2/update_main/q1/dense_1/bias/ApplyAdam/^Adam_2/update_main/q1/dense_1/kernel/ApplyAdam-^Adam_2/update_main/q1/dense_2/bias/ApplyAdam/^Adam_2/update_main/q1/dense_2/kernel/ApplyAdam+^Adam_2/update_main/q2/dense/bias/ApplyAdam-^Adam_2/update_main/q2/dense/kernel/ApplyAdam-^Adam_2/update_main/q2/dense_1/bias/ApplyAdam/^Adam_2/update_main/q2/dense_1/kernel/ApplyAdam-^Adam_2/update_main/q2/dense_2/bias/ApplyAdam/^Adam_2/update_main/q2/dense_2/kernel/ApplyAdam*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
�
Adam_2/AssignAssignbeta1_power_2
Adam_2/mul*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking( 
�
Adam_2/mul_1Mulbeta2_power_2/readAdam_2/beta2+^Adam_2/update_main/q1/dense/bias/ApplyAdam-^Adam_2/update_main/q1/dense/kernel/ApplyAdam-^Adam_2/update_main/q1/dense_1/bias/ApplyAdam/^Adam_2/update_main/q1/dense_1/kernel/ApplyAdam-^Adam_2/update_main/q1/dense_2/bias/ApplyAdam/^Adam_2/update_main/q1/dense_2/kernel/ApplyAdam+^Adam_2/update_main/q2/dense/bias/ApplyAdam-^Adam_2/update_main/q2/dense/kernel/ApplyAdam-^Adam_2/update_main/q2/dense_1/bias/ApplyAdam/^Adam_2/update_main/q2/dense_1/kernel/ApplyAdam-^Adam_2/update_main/q2/dense_2/bias/ApplyAdam/^Adam_2/update_main/q2/dense_2/kernel/ApplyAdam*
T0*%
_class
loc:@main/q1/dense/bias*
_output_shapes
: 
�
Adam_2/Assign_1Assignbeta2_power_2Adam_2/mul_1*
validate_shape(*
_output_shapes
: *
use_locking( *
T0*%
_class
loc:@main/q1/dense/bias
�
Adam_2NoOp^Adam_1^Adam_2/Assign^Adam_2/Assign_1+^Adam_2/update_main/q1/dense/bias/ApplyAdam-^Adam_2/update_main/q1/dense/kernel/ApplyAdam-^Adam_2/update_main/q1/dense_1/bias/ApplyAdam/^Adam_2/update_main/q1/dense_1/kernel/ApplyAdam-^Adam_2/update_main/q1/dense_2/bias/ApplyAdam/^Adam_2/update_main/q1/dense_2/kernel/ApplyAdam+^Adam_2/update_main/q2/dense/bias/ApplyAdam-^Adam_2/update_main/q2/dense/kernel/ApplyAdam-^Adam_2/update_main/q2/dense_1/bias/ApplyAdam/^Adam_2/update_main/q2/dense_1/kernel/ApplyAdam-^Adam_2/update_main/q2/dense_2/bias/ApplyAdam/^Adam_2/update_main/q2/dense_2/kernel/ApplyAdam
]
gradients_3/ShapeConst^Adam_2*
valueB *
dtype0*
_output_shapes
: 
c
gradients_3/grad_ys_0Const^Adam_2*
_output_shapes
: *
valueB
 *  �?*
dtype0
u
gradients_3/FillFillgradients_3/Shapegradients_3/grad_ys_0*

index_type0*
_output_shapes
: *
T0
T
gradients_3/Neg_2_grad/NegNeggradients_3/Fill*
_output_shapes
: *
T0
x
%gradients_3/Mean_3_grad/Reshape/shapeConst^Adam_2*
valueB:*
dtype0*
_output_shapes
:
�
gradients_3/Mean_3_grad/ReshapeReshapegradients_3/Neg_2_grad/Neg%gradients_3/Mean_3_grad/Reshape/shape*
T0*
Tshape0*
_output_shapes
:
l
gradients_3/Mean_3_grad/ShapeShapemul_16^Adam_2*
T0*
out_type0*
_output_shapes
:
�
gradients_3/Mean_3_grad/TileTilegradients_3/Mean_3_grad/Reshapegradients_3/Mean_3_grad/Shape*#
_output_shapes
:���������*

Tmultiples0*
T0
n
gradients_3/Mean_3_grad/Shape_1Shapemul_16^Adam_2*
out_type0*
_output_shapes
:*
T0
k
gradients_3/Mean_3_grad/Shape_2Const^Adam_2*
_output_shapes
: *
valueB *
dtype0
p
gradients_3/Mean_3_grad/ConstConst^Adam_2*
valueB: *
dtype0*
_output_shapes
:
�
gradients_3/Mean_3_grad/ProdProdgradients_3/Mean_3_grad/Shape_1gradients_3/Mean_3_grad/Const*
T0*
_output_shapes
: *
	keep_dims( *

Tidx0
r
gradients_3/Mean_3_grad/Const_1Const^Adam_2*
valueB: *
dtype0*
_output_shapes
:
�
gradients_3/Mean_3_grad/Prod_1Prodgradients_3/Mean_3_grad/Shape_2gradients_3/Mean_3_grad/Const_1*
	keep_dims( *

Tidx0*
T0*
_output_shapes
: 
l
!gradients_3/Mean_3_grad/Maximum/yConst^Adam_2*
value	B :*
dtype0*
_output_shapes
: 
�
gradients_3/Mean_3_grad/MaximumMaximumgradients_3/Mean_3_grad/Prod_1!gradients_3/Mean_3_grad/Maximum/y*
T0*
_output_shapes
: 
�
 gradients_3/Mean_3_grad/floordivFloorDivgradients_3/Mean_3_grad/Prodgradients_3/Mean_3_grad/Maximum*
_output_shapes
: *
T0
�
gradients_3/Mean_3_grad/CastCast gradients_3/Mean_3_grad/floordiv*

SrcT0*
Truncate( *
_output_shapes
: *

DstT0
�
gradients_3/Mean_3_grad/truedivRealDivgradients_3/Mean_3_grad/Tilegradients_3/Mean_3_grad/Cast*
T0*#
_output_shapes
:���������
r
gradients_3/mul_16_grad/ShapeShapelog_alpha/read^Adam_2*
_output_shapes
: *
T0*
out_type0
v
gradients_3/mul_16_grad/Shape_1ShapeStopGradient_1^Adam_2*
T0*
out_type0*
_output_shapes
:
�
-gradients_3/mul_16_grad/BroadcastGradientArgsBroadcastGradientArgsgradients_3/mul_16_grad/Shapegradients_3/mul_16_grad/Shape_1*
T0*2
_output_shapes 
:���������:���������
�
gradients_3/mul_16_grad/MulMulgradients_3/Mean_3_grad/truedivStopGradient_1*#
_output_shapes
:���������*
T0
�
gradients_3/mul_16_grad/SumSumgradients_3/mul_16_grad/Mul-gradients_3/mul_16_grad/BroadcastGradientArgs*
_output_shapes
:*
	keep_dims( *

Tidx0*
T0
�
gradients_3/mul_16_grad/ReshapeReshapegradients_3/mul_16_grad/Sumgradients_3/mul_16_grad/Shape*
Tshape0*
_output_shapes
: *
T0
�
gradients_3/mul_16_grad/Mul_1Mullog_alpha/readgradients_3/Mean_3_grad/truediv*#
_output_shapes
:���������*
T0
�
gradients_3/mul_16_grad/Sum_1Sumgradients_3/mul_16_grad/Mul_1/gradients_3/mul_16_grad/BroadcastGradientArgs:1*
T0*
_output_shapes
:*
	keep_dims( *

Tidx0
�
!gradients_3/mul_16_grad/Reshape_1Reshapegradients_3/mul_16_grad/Sum_1gradients_3/mul_16_grad/Shape_1*
Tshape0*#
_output_shapes
:���������*
T0

(gradients_3/mul_16_grad/tuple/group_depsNoOp^Adam_2 ^gradients_3/mul_16_grad/Reshape"^gradients_3/mul_16_grad/Reshape_1
�
0gradients_3/mul_16_grad/tuple/control_dependencyIdentitygradients_3/mul_16_grad/Reshape)^gradients_3/mul_16_grad/tuple/group_deps*
T0*2
_class(
&$loc:@gradients_3/mul_16_grad/Reshape*
_output_shapes
: 
�
2gradients_3/mul_16_grad/tuple/control_dependency_1Identity!gradients_3/mul_16_grad/Reshape_1)^gradients_3/mul_16_grad/tuple/group_deps*
T0*4
_class*
(&loc:@gradients_3/mul_16_grad/Reshape_1*#
_output_shapes
:���������
~
beta1_power_3/initial_valueConst*
valueB
 *fff?*
_class
loc:@log_alpha*
dtype0*
_output_shapes
: 
�
beta1_power_3
VariableV2*
shared_name *
_class
loc:@log_alpha*
	container *
shape: *
dtype0*
_output_shapes
: 
�
beta1_power_3/AssignAssignbeta1_power_3beta1_power_3/initial_value*
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: 
l
beta1_power_3/readIdentitybeta1_power_3*
T0*
_class
loc:@log_alpha*
_output_shapes
: 
~
beta2_power_3/initial_valueConst*
dtype0*
_output_shapes
: *
valueB
 *w�?*
_class
loc:@log_alpha
�
beta2_power_3
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@log_alpha*
	container 
�
beta2_power_3/AssignAssignbeta2_power_3beta2_power_3/initial_value*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(
l
beta2_power_3/readIdentitybeta2_power_3*
_output_shapes
: *
T0*
_class
loc:@log_alpha
�
 log_alpha/Adam/Initializer/zerosConst*
_class
loc:@log_alpha*
valueB
 *    *
dtype0*
_output_shapes
: 
�
log_alpha/Adam
VariableV2*
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@log_alpha*
	container 
�
log_alpha/Adam/AssignAssignlog_alpha/Adam log_alpha/Adam/Initializer/zeros*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
n
log_alpha/Adam/readIdentitylog_alpha/Adam*
_output_shapes
: *
T0*
_class
loc:@log_alpha
�
"log_alpha/Adam_1/Initializer/zerosConst*
_class
loc:@log_alpha*
valueB
 *    *
dtype0*
_output_shapes
: 
�
log_alpha/Adam_1
VariableV2*
	container *
shape: *
dtype0*
_output_shapes
: *
shared_name *
_class
loc:@log_alpha
�
log_alpha/Adam_1/AssignAssignlog_alpha/Adam_1"log_alpha/Adam_1/Initializer/zeros*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: *
use_locking(
r
log_alpha/Adam_1/readIdentitylog_alpha/Adam_1*
T0*
_class
loc:@log_alpha*
_output_shapes
: 
b
Adam_3/learning_rateConst^Adam_2*
valueB
 *RI�9*
dtype0*
_output_shapes
: 
Z
Adam_3/beta1Const^Adam_2*
valueB
 *fff?*
dtype0*
_output_shapes
: 
Z
Adam_3/beta2Const^Adam_2*
valueB
 *w�?*
dtype0*
_output_shapes
: 
\
Adam_3/epsilonConst^Adam_2*
valueB
 *��8*
dtype0*
_output_shapes
: 
�
!Adam_3/update_log_alpha/ApplyAdam	ApplyAdam	log_alphalog_alpha/Adamlog_alpha/Adam_1beta1_power_3/readbeta2_power_3/readAdam_3/learning_rateAdam_3/beta1Adam_3/beta2Adam_3/epsilon0gradients_3/mul_16_grad/tuple/control_dependency*
_output_shapes
: *
use_locking( *
T0*
_class
loc:@log_alpha*
use_nesterov( 
�

Adam_3/mulMulbeta1_power_3/readAdam_3/beta1"^Adam_3/update_log_alpha/ApplyAdam*
T0*
_class
loc:@log_alpha*
_output_shapes
: 
�
Adam_3/AssignAssignbeta1_power_3
Adam_3/mul*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: *
use_locking( *
T0
�
Adam_3/mul_1Mulbeta2_power_3/readAdam_3/beta2"^Adam_3/update_log_alpha/ApplyAdam*
T0*
_class
loc:@log_alpha*
_output_shapes
: 
�
Adam_3/Assign_1Assignbeta2_power_3Adam_3/mul_1*
use_locking( *
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: 
]
Adam_3NoOp^Adam_2^Adam_3/Assign^Adam_3/Assign_1"^Adam_3/update_log_alpha/ApplyAdam
V
mul_17/xConst^Adam_2*
_output_shapes
: *
valueB
 *R�~?*
dtype0
]
mul_17Mulmul_17/xtarget/pi/dense/kernel/read*
T0*
_output_shapes

:@
V
mul_18/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
[
mul_18Mulmul_18/xmain/pi/dense/kernel/read*
T0*
_output_shapes

:@
G
add_6AddV2mul_17mul_18*
T0*
_output_shapes

:@
�
Assign_1Assigntarget/pi/dense/kerneladd_6*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
V
mul_19/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
W
mul_19Mulmul_19/xtarget/pi/dense/bias/read*
T0*
_output_shapes
:@
V
mul_20/xConst^Adam_2*
_output_shapes
: *
valueB
 *
ף;*
dtype0
U
mul_20Mulmul_20/xmain/pi/dense/bias/read*
_output_shapes
:@*
T0
C
add_7AddV2mul_19mul_20*
T0*
_output_shapes
:@
�
Assign_2Assigntarget/pi/dense/biasadd_7*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(
V
mul_21/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
_
mul_21Mulmul_21/xtarget/pi/dense_1/kernel/read*
T0*
_output_shapes

:@@
V
mul_22/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
]
mul_22Mulmul_22/xmain/pi/dense_1/kernel/read*
T0*
_output_shapes

:@@
G
add_8AddV2mul_21mul_22*
_output_shapes

:@@*
T0
�
Assign_3Assigntarget/pi/dense_1/kerneladd_8*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
V
mul_23/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
Y
mul_23Mulmul_23/xtarget/pi/dense_1/bias/read*
T0*
_output_shapes
:@
V
mul_24/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
W
mul_24Mulmul_24/xmain/pi/dense_1/bias/read*
_output_shapes
:@*
T0
C
add_9AddV2mul_23mul_24*
_output_shapes
:@*
T0
�
Assign_4Assigntarget/pi/dense_1/biasadd_9*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
V
mul_25/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
_
mul_25Mulmul_25/xtarget/pi/dense_2/kernel/read*
T0*
_output_shapes

:@
V
mul_26/xConst^Adam_2*
dtype0*
_output_shapes
: *
valueB
 *
ף;
]
mul_26Mulmul_26/xmain/pi/dense_2/kernel/read*
_output_shapes

:@*
T0
H
add_10AddV2mul_25mul_26*
_output_shapes

:@*
T0
�
Assign_5Assigntarget/pi/dense_2/kerneladd_10*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
V
mul_27/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
Y
mul_27Mulmul_27/xtarget/pi/dense_2/bias/read*
T0*
_output_shapes
:
V
mul_28/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
W
mul_28Mulmul_28/xmain/pi/dense_2/bias/read*
T0*
_output_shapes
:
D
add_11AddV2mul_27mul_28*
_output_shapes
:*
T0
�
Assign_6Assigntarget/pi/dense_2/biasadd_11*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
V
mul_29/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
]
mul_29Mulmul_29/xtarget/q1/dense/kernel/read*
_output_shapes

:@*
T0
V
mul_30/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
[
mul_30Mulmul_30/xmain/q1/dense/kernel/read*
_output_shapes

:@*
T0
H
add_12AddV2mul_29mul_30*
T0*
_output_shapes

:@
�
Assign_7Assigntarget/q1/dense/kerneladd_12*
T0*)
_class
loc:@target/q1/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
V
mul_31/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
W
mul_31Mulmul_31/xtarget/q1/dense/bias/read*
_output_shapes
:@*
T0
V
mul_32/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
U
mul_32Mulmul_32/xmain/q1/dense/bias/read*
_output_shapes
:@*
T0
D
add_13AddV2mul_31mul_32*
T0*
_output_shapes
:@
�
Assign_8Assigntarget/q1/dense/biasadd_13*
use_locking(*
T0*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes
:@
V
mul_33/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
_
mul_33Mulmul_33/xtarget/q1/dense_1/kernel/read*
T0*
_output_shapes

:@@
V
mul_34/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
]
mul_34Mulmul_34/xmain/q1/dense_1/kernel/read*
_output_shapes

:@@*
T0
H
add_14AddV2mul_33mul_34*
T0*
_output_shapes

:@@
�
Assign_9Assigntarget/q1/dense_1/kerneladd_14*
_output_shapes

:@@*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(
V
mul_35/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
Y
mul_35Mulmul_35/xtarget/q1/dense_1/bias/read*
T0*
_output_shapes
:@
V
mul_36/xConst^Adam_2*
dtype0*
_output_shapes
: *
valueB
 *
ף;
W
mul_36Mulmul_36/xmain/q1/dense_1/bias/read*
T0*
_output_shapes
:@
D
add_15AddV2mul_35mul_36*
T0*
_output_shapes
:@
�
	Assign_10Assigntarget/q1/dense_1/biasadd_15*
use_locking(*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
V
mul_37/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
_
mul_37Mulmul_37/xtarget/q1/dense_2/kernel/read*
_output_shapes

:@*
T0
V
mul_38/xConst^Adam_2*
_output_shapes
: *
valueB
 *
ף;*
dtype0
]
mul_38Mulmul_38/xmain/q1/dense_2/kernel/read*
_output_shapes

:@*
T0
H
add_16AddV2mul_37mul_38*
T0*
_output_shapes

:@
�
	Assign_11Assigntarget/q1/dense_2/kerneladd_16*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@
V
mul_39/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
Y
mul_39Mulmul_39/xtarget/q1/dense_2/bias/read*
T0*
_output_shapes
:
V
mul_40/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
W
mul_40Mulmul_40/xmain/q1/dense_2/bias/read*
_output_shapes
:*
T0
D
add_17AddV2mul_39mul_40*
_output_shapes
:*
T0
�
	Assign_12Assigntarget/q1/dense_2/biasadd_17*
T0*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
V
mul_41/xConst^Adam_2*
dtype0*
_output_shapes
: *
valueB
 *R�~?
]
mul_41Mulmul_41/xtarget/q2/dense/kernel/read*
T0*
_output_shapes

:@
V
mul_42/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
[
mul_42Mulmul_42/xmain/q2/dense/kernel/read*
T0*
_output_shapes

:@
H
add_18AddV2mul_41mul_42*
T0*
_output_shapes

:@
�
	Assign_13Assigntarget/q2/dense/kerneladd_18*
T0*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
V
mul_43/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
W
mul_43Mulmul_43/xtarget/q2/dense/bias/read*
_output_shapes
:@*
T0
V
mul_44/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
U
mul_44Mulmul_44/xmain/q2/dense/bias/read*
_output_shapes
:@*
T0
D
add_19AddV2mul_43mul_44*
T0*
_output_shapes
:@
�
	Assign_14Assigntarget/q2/dense/biasadd_19*
use_locking(*
T0*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes
:@
V
mul_45/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
_
mul_45Mulmul_45/xtarget/q2/dense_1/kernel/read*
_output_shapes

:@@*
T0
V
mul_46/xConst^Adam_2*
_output_shapes
: *
valueB
 *
ף;*
dtype0
]
mul_46Mulmul_46/xmain/q2/dense_1/kernel/read*
T0*
_output_shapes

:@@
H
add_20AddV2mul_45mul_46*
_output_shapes

:@@*
T0
�
	Assign_15Assigntarget/q2/dense_1/kerneladd_20*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
V
mul_47/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
Y
mul_47Mulmul_47/xtarget/q2/dense_1/bias/read*
T0*
_output_shapes
:@
V
mul_48/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
W
mul_48Mulmul_48/xmain/q2/dense_1/bias/read*
T0*
_output_shapes
:@
D
add_21AddV2mul_47mul_48*
T0*
_output_shapes
:@
�
	Assign_16Assigntarget/q2/dense_1/biasadd_21*
use_locking(*
T0*)
_class
loc:@target/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
V
mul_49/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
_
mul_49Mulmul_49/xtarget/q2/dense_2/kernel/read*
_output_shapes

:@*
T0
V
mul_50/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
]
mul_50Mulmul_50/xmain/q2/dense_2/kernel/read*
T0*
_output_shapes

:@
H
add_22AddV2mul_49mul_50*
T0*
_output_shapes

:@
�
	Assign_17Assigntarget/q2/dense_2/kerneladd_22*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_2/kernel
V
mul_51/xConst^Adam_2*
valueB
 *R�~?*
dtype0*
_output_shapes
: 
Y
mul_51Mulmul_51/xtarget/q2/dense_2/bias/read*
T0*
_output_shapes
:
V
mul_52/xConst^Adam_2*
valueB
 *
ף;*
dtype0*
_output_shapes
: 
W
mul_52Mulmul_52/xmain/q2/dense_2/bias/read*
_output_shapes
:*
T0
D
add_23AddV2mul_51mul_52*
T0*
_output_shapes
:
�
	Assign_18Assigntarget/q2/dense_2/biasadd_23*
use_locking(*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�

group_depsNoOp^Adam_2	^Assign_1
^Assign_10
^Assign_11
^Assign_12
^Assign_13
^Assign_14
^Assign_15
^Assign_16
^Assign_17
^Assign_18	^Assign_2	^Assign_3	^Assign_4	^Assign_5	^Assign_6	^Assign_7	^Assign_8	^Assign_9
�
	Assign_19Assigntarget/pi/dense/kernelmain/pi/dense/kernel/read*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
	Assign_20Assigntarget/pi/dense/biasmain/pi/dense/bias/read*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
	Assign_21Assigntarget/pi/dense_1/kernelmain/pi/dense_1/kernel/read*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel
�
	Assign_22Assigntarget/pi/dense_1/biasmain/pi/dense_1/bias/read*
_output_shapes
:@*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(
�
	Assign_23Assigntarget/pi/dense_2/kernelmain/pi/dense_2/kernel/read*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
	Assign_24Assigntarget/pi/dense_2/biasmain/pi/dense_2/bias/read*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
	Assign_25Assigntarget/q1/dense/kernelmain/q1/dense/kernel/read*
use_locking(*
T0*)
_class
loc:@target/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
	Assign_26Assigntarget/q1/dense/biasmain/q1/dense/bias/read*
use_locking(*
T0*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes
:@
�
	Assign_27Assigntarget/q1/dense_1/kernelmain/q1/dense_1/kernel/read*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
	Assign_28Assigntarget/q1/dense_1/biasmain/q1/dense_1/bias/read*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*)
_class
loc:@target/q1/dense_1/bias
�
	Assign_29Assigntarget/q1/dense_2/kernelmain/q1/dense_2/kernel/read*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_2/kernel
�
	Assign_30Assigntarget/q1/dense_2/biasmain/q1/dense_2/bias/read*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*)
_class
loc:@target/q1/dense_2/bias
�
	Assign_31Assigntarget/q2/dense/kernelmain/q2/dense/kernel/read*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
	Assign_32Assigntarget/q2/dense/biasmain/q2/dense/bias/read*
use_locking(*
T0*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
	Assign_33Assigntarget/q2/dense_1/kernelmain/q2/dense_1/kernel/read*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
	Assign_34Assigntarget/q2/dense_1/biasmain/q2/dense_1/bias/read*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*)
_class
loc:@target/q2/dense_1/bias
�
	Assign_35Assigntarget/q2/dense_2/kernelmain/q2/dense_2/kernel/read*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
	Assign_36Assigntarget/q2/dense_2/biasmain/q2/dense_2/bias/read*
use_locking(*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
group_deps_1NoOp
^Assign_19
^Assign_20
^Assign_21
^Assign_22
^Assign_23
^Assign_24
^Assign_25
^Assign_26
^Assign_27
^Assign_28
^Assign_29
^Assign_30
^Assign_31
^Assign_32
^Assign_33
^Assign_34
^Assign_35
^Assign_36
�
initNoOp^R/Adam/Assign^R/Adam_1/Assign	^R/Assign^beta1_power/Assign^beta1_power_1/Assign^beta1_power_2/Assign^beta1_power_3/Assign^beta2_power/Assign^beta2_power_1/Assign^beta2_power_2/Assign^beta2_power_3/Assign^log_alpha/Adam/Assign^log_alpha/Adam_1/Assign^log_alpha/Assign^main/pi/dense/bias/Adam/Assign!^main/pi/dense/bias/Adam_1/Assign^main/pi/dense/bias/Assign!^main/pi/dense/kernel/Adam/Assign#^main/pi/dense/kernel/Adam_1/Assign^main/pi/dense/kernel/Assign!^main/pi/dense_1/bias/Adam/Assign#^main/pi/dense_1/bias/Adam_1/Assign^main/pi/dense_1/bias/Assign#^main/pi/dense_1/kernel/Adam/Assign%^main/pi/dense_1/kernel/Adam_1/Assign^main/pi/dense_1/kernel/Assign!^main/pi/dense_2/bias/Adam/Assign#^main/pi/dense_2/bias/Adam_1/Assign^main/pi/dense_2/bias/Assign#^main/pi/dense_2/kernel/Adam/Assign%^main/pi/dense_2/kernel/Adam_1/Assign^main/pi/dense_2/kernel/Assign^main/q1/dense/bias/Adam/Assign!^main/q1/dense/bias/Adam_1/Assign^main/q1/dense/bias/Assign!^main/q1/dense/kernel/Adam/Assign#^main/q1/dense/kernel/Adam_1/Assign^main/q1/dense/kernel/Assign!^main/q1/dense_1/bias/Adam/Assign#^main/q1/dense_1/bias/Adam_1/Assign^main/q1/dense_1/bias/Assign#^main/q1/dense_1/kernel/Adam/Assign%^main/q1/dense_1/kernel/Adam_1/Assign^main/q1/dense_1/kernel/Assign!^main/q1/dense_2/bias/Adam/Assign#^main/q1/dense_2/bias/Adam_1/Assign^main/q1/dense_2/bias/Assign#^main/q1/dense_2/kernel/Adam/Assign%^main/q1/dense_2/kernel/Adam_1/Assign^main/q1/dense_2/kernel/Assign^main/q2/dense/bias/Adam/Assign!^main/q2/dense/bias/Adam_1/Assign^main/q2/dense/bias/Assign!^main/q2/dense/kernel/Adam/Assign#^main/q2/dense/kernel/Adam_1/Assign^main/q2/dense/kernel/Assign!^main/q2/dense_1/bias/Adam/Assign#^main/q2/dense_1/bias/Adam_1/Assign^main/q2/dense_1/bias/Assign#^main/q2/dense_1/kernel/Adam/Assign%^main/q2/dense_1/kernel/Adam_1/Assign^main/q2/dense_1/kernel/Assign!^main/q2/dense_2/bias/Adam/Assign#^main/q2/dense_2/bias/Adam_1/Assign^main/q2/dense_2/bias/Assign#^main/q2/dense_2/kernel/Adam/Assign%^main/q2/dense_2/kernel/Adam_1/Assign^main/q2/dense_2/kernel/Assign^target/pi/dense/bias/Assign^target/pi/dense/kernel/Assign^target/pi/dense_1/bias/Assign ^target/pi/dense_1/kernel/Assign^target/pi/dense_2/bias/Assign ^target/pi/dense_2/kernel/Assign^target/q1/dense/bias/Assign^target/q1/dense/kernel/Assign^target/q1/dense_1/bias/Assign ^target/q1/dense_1/kernel/Assign^target/q1/dense_2/bias/Assign ^target/q1/dense_2/kernel/Assign^target/q2/dense/bias/Assign^target/q2/dense/kernel/Assign^target/q2/dense_1/bias/Assign ^target/q2/dense_1/kernel/Assign^target/q2/dense_2/bias/Assign ^target/q2/dense_2/kernel/Assign
Y
save/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
n
save/filenamePlaceholderWithDefaultsave/filename/input*
dtype0*
_output_shapes
: *
shape: 
e

save/ConstPlaceholderWithDefaultsave/filename*
dtype0*
_output_shapes
: *
shape: 
�
save/StringJoin/inputs_1Const*<
value3B1 B+_temp_8a0d07de498648f9b7a4642472bc3c48/part*
dtype0*
_output_shapes
: 
u
save/StringJoin
StringJoin
save/Constsave/StringJoin/inputs_1*
N*
_output_shapes
: *
	separator 
Q
save/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
\
save/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
}
save/ShardedFilenameShardedFilenamesave/StringJoinsave/ShardedFilename/shardsave/num_shards*
_output_shapes
: 
�
save/SaveV2/tensor_namesConst*�
value�B�VBRBR/AdamBR/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3B	log_alphaBlog_alpha/AdamBlog_alpha/Adam_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel*
dtype0*
_output_shapes
:V
�
save/SaveV2/shape_and_slicesConst*�
value�B�VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:V
�
save/SaveV2SaveV2save/ShardedFilenamesave/SaveV2/tensor_namessave/SaveV2/shape_and_slicesRR/AdamR/Adam_1beta1_powerbeta1_power_1beta1_power_2beta1_power_3beta2_powerbeta2_power_1beta2_power_2beta2_power_3	log_alphalog_alpha/Adamlog_alpha/Adam_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1main/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1main/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1main/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1main/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1main/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1main/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1main/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1main/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1main/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1main/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1main/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q1/dense/biastarget/q1/dense/kerneltarget/q1/dense_1/biastarget/q1/dense_1/kerneltarget/q1/dense_2/biastarget/q1/dense_2/kerneltarget/q2/dense/biastarget/q2/dense/kerneltarget/q2/dense_1/biastarget/q2/dense_1/kerneltarget/q2/dense_2/biastarget/q2/dense_2/kernel*d
dtypesZ
X2V
�
save/control_dependencyIdentitysave/ShardedFilename^save/SaveV2*
T0*'
_class
loc:@save/ShardedFilename*
_output_shapes
: 
�
+save/MergeV2Checkpoints/checkpoint_prefixesPacksave/ShardedFilename^save/control_dependency*

axis *
N*
_output_shapes
:*
T0
}
save/MergeV2CheckpointsMergeV2Checkpoints+save/MergeV2Checkpoints/checkpoint_prefixes
save/Const*
delete_old_dirs(
z
save/IdentityIdentity
save/Const^save/MergeV2Checkpoints^save/control_dependency*
T0*
_output_shapes
: 
�
save/RestoreV2/tensor_namesConst*�
value�B�VBRBR/AdamBR/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3B	log_alphaBlog_alpha/AdamBlog_alpha/Adam_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel*
dtype0*
_output_shapes
:V
�
save/RestoreV2/shape_and_slicesConst*�
value�B�VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:V
�
save/RestoreV2	RestoreV2
save/Constsave/RestoreV2/tensor_namessave/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*d
dtypesZ
X2V
�
save/AssignAssignRsave/RestoreV2*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes

:
�
save/Assign_1AssignR/Adamsave/RestoreV2:1*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes

:*
use_locking(
�
save/Assign_2AssignR/Adam_1save/RestoreV2:2*
_class

loc:@R*
validate_shape(*
_output_shapes

:*
use_locking(*
T0
�
save/Assign_3Assignbeta1_powersave/RestoreV2:3*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: 
�
save/Assign_4Assignbeta1_power_1save/RestoreV2:4*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
save/Assign_5Assignbeta1_power_2save/RestoreV2:5*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save/Assign_6Assignbeta1_power_3save/RestoreV2:6*
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: 
�
save/Assign_7Assignbeta2_powersave/RestoreV2:7*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: 
�
save/Assign_8Assignbeta2_power_1save/RestoreV2:8*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: 
�
save/Assign_9Assignbeta2_power_2save/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias
�
save/Assign_10Assignbeta2_power_3save/RestoreV2:10*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(
�
save/Assign_11Assign	log_alphasave/RestoreV2:11*
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: 
�
save/Assign_12Assignlog_alpha/Adamsave/RestoreV2:12*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(
�
save/Assign_13Assignlog_alpha/Adam_1save/RestoreV2:13*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(
�
save/Assign_14Assignmain/pi/dense/biassave/RestoreV2:14*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save/Assign_15Assignmain/pi/dense/bias/Adamsave/RestoreV2:15*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
�
save/Assign_16Assignmain/pi/dense/bias/Adam_1save/RestoreV2:16*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_17Assignmain/pi/dense/kernelsave/RestoreV2:17*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save/Assign_18Assignmain/pi/dense/kernel/Adamsave/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_19Assignmain/pi/dense/kernel/Adam_1save/RestoreV2:19*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save/Assign_20Assignmain/pi/dense_1/biassave/RestoreV2:20*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save/Assign_21Assignmain/pi/dense_1/bias/Adamsave/RestoreV2:21*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_22Assignmain/pi/dense_1/bias/Adam_1save/RestoreV2:22*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_23Assignmain/pi/dense_1/kernelsave/RestoreV2:23*
_output_shapes

:@@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(
�
save/Assign_24Assignmain/pi/dense_1/kernel/Adamsave/RestoreV2:24*
_output_shapes

:@@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(
�
save/Assign_25Assignmain/pi/dense_1/kernel/Adam_1save/RestoreV2:25*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save/Assign_26Assignmain/pi/dense_2/biassave/RestoreV2:26*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_27Assignmain/pi/dense_2/bias/Adamsave/RestoreV2:27*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(
�
save/Assign_28Assignmain/pi/dense_2/bias/Adam_1save/RestoreV2:28*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_29Assignmain/pi/dense_2/kernelsave/RestoreV2:29*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(
�
save/Assign_30Assignmain/pi/dense_2/kernel/Adamsave/RestoreV2:30*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_31Assignmain/pi/dense_2/kernel/Adam_1save/RestoreV2:31*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(
�
save/Assign_32Assignmain/q1/dense/biassave/RestoreV2:32*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(
�
save/Assign_33Assignmain/q1/dense/bias/Adamsave/RestoreV2:33*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save/Assign_34Assignmain/q1/dense/bias/Adam_1save/RestoreV2:34*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_35Assignmain/q1/dense/kernelsave/RestoreV2:35*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_36Assignmain/q1/dense/kernel/Adamsave/RestoreV2:36*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_37Assignmain/q1/dense/kernel/Adam_1save/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(
�
save/Assign_38Assignmain/q1/dense_1/biassave/RestoreV2:38*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_39Assignmain/q1/dense_1/bias/Adamsave/RestoreV2:39*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(
�
save/Assign_40Assignmain/q1/dense_1/bias/Adam_1save/RestoreV2:40*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_41Assignmain/q1/dense_1/kernelsave/RestoreV2:41*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save/Assign_42Assignmain/q1/dense_1/kernel/Adamsave/RestoreV2:42*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save/Assign_43Assignmain/q1/dense_1/kernel/Adam_1save/RestoreV2:43*
_output_shapes

:@@*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(
�
save/Assign_44Assignmain/q1/dense_2/biassave/RestoreV2:44*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_45Assignmain/q1/dense_2/bias/Adamsave/RestoreV2:45*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_46Assignmain/q1/dense_2/bias/Adam_1save/RestoreV2:46*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_47Assignmain/q1/dense_2/kernelsave/RestoreV2:47*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel
�
save/Assign_48Assignmain/q1/dense_2/kernel/Adamsave/RestoreV2:48*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_49Assignmain/q1/dense_2/kernel/Adam_1save/RestoreV2:49*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_50Assignmain/q2/dense/biassave/RestoreV2:50*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save/Assign_51Assignmain/q2/dense/bias/Adamsave/RestoreV2:51*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_52Assignmain/q2/dense/bias/Adam_1save/RestoreV2:52*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_53Assignmain/q2/dense/kernelsave/RestoreV2:53*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_54Assignmain/q2/dense/kernel/Adamsave/RestoreV2:54*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_55Assignmain/q2/dense/kernel/Adam_1save/RestoreV2:55*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save/Assign_56Assignmain/q2/dense_1/biassave/RestoreV2:56*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_57Assignmain/q2/dense_1/bias/Adamsave/RestoreV2:57*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_58Assignmain/q2/dense_1/bias/Adam_1save/RestoreV2:58*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_59Assignmain/q2/dense_1/kernelsave/RestoreV2:59*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save/Assign_60Assignmain/q2/dense_1/kernel/Adamsave/RestoreV2:60*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save/Assign_61Assignmain/q2/dense_1/kernel/Adam_1save/RestoreV2:61*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save/Assign_62Assignmain/q2/dense_2/biassave/RestoreV2:62*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_63Assignmain/q2/dense_2/bias/Adamsave/RestoreV2:63*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_64Assignmain/q2/dense_2/bias/Adam_1save/RestoreV2:64*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save/Assign_65Assignmain/q2/dense_2/kernelsave/RestoreV2:65*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel
�
save/Assign_66Assignmain/q2/dense_2/kernel/Adamsave/RestoreV2:66*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_67Assignmain/q2/dense_2/kernel/Adam_1save/RestoreV2:67*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_68Assigntarget/pi/dense/biassave/RestoreV2:68*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_69Assigntarget/pi/dense/kernelsave/RestoreV2:69*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save/Assign_70Assigntarget/pi/dense_1/biassave/RestoreV2:70*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_71Assigntarget/pi/dense_1/kernelsave/RestoreV2:71*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save/Assign_72Assigntarget/pi/dense_2/biassave/RestoreV2:72*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_73Assigntarget/pi/dense_2/kernelsave/RestoreV2:73*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel
�
save/Assign_74Assigntarget/q1/dense/biassave/RestoreV2:74*
use_locking(*
T0*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_75Assigntarget/q1/dense/kernelsave/RestoreV2:75*)
_class
loc:@target/q1/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save/Assign_76Assigntarget/q1/dense_1/biassave/RestoreV2:76*
use_locking(*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_77Assigntarget/q1/dense_1/kernelsave/RestoreV2:77*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save/Assign_78Assigntarget/q1/dense_2/biassave/RestoreV2:78*
T0*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save/Assign_79Assigntarget/q1/dense_2/kernelsave/RestoreV2:79*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save/Assign_80Assigntarget/q2/dense/biassave/RestoreV2:80*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save/Assign_81Assigntarget/q2/dense/kernelsave/RestoreV2:81*
use_locking(*
T0*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save/Assign_82Assigntarget/q2/dense_1/biassave/RestoreV2:82*
use_locking(*
T0*)
_class
loc:@target/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save/Assign_83Assigntarget/q2/dense_1/kernelsave/RestoreV2:83*
_output_shapes

:@@*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(
�
save/Assign_84Assigntarget/q2/dense_2/biassave/RestoreV2:84*
use_locking(*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save/Assign_85Assigntarget/q2/dense_2/kernelsave/RestoreV2:85*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save/restore_shardNoOp^save/Assign^save/Assign_1^save/Assign_10^save/Assign_11^save/Assign_12^save/Assign_13^save/Assign_14^save/Assign_15^save/Assign_16^save/Assign_17^save/Assign_18^save/Assign_19^save/Assign_2^save/Assign_20^save/Assign_21^save/Assign_22^save/Assign_23^save/Assign_24^save/Assign_25^save/Assign_26^save/Assign_27^save/Assign_28^save/Assign_29^save/Assign_3^save/Assign_30^save/Assign_31^save/Assign_32^save/Assign_33^save/Assign_34^save/Assign_35^save/Assign_36^save/Assign_37^save/Assign_38^save/Assign_39^save/Assign_4^save/Assign_40^save/Assign_41^save/Assign_42^save/Assign_43^save/Assign_44^save/Assign_45^save/Assign_46^save/Assign_47^save/Assign_48^save/Assign_49^save/Assign_5^save/Assign_50^save/Assign_51^save/Assign_52^save/Assign_53^save/Assign_54^save/Assign_55^save/Assign_56^save/Assign_57^save/Assign_58^save/Assign_59^save/Assign_6^save/Assign_60^save/Assign_61^save/Assign_62^save/Assign_63^save/Assign_64^save/Assign_65^save/Assign_66^save/Assign_67^save/Assign_68^save/Assign_69^save/Assign_7^save/Assign_70^save/Assign_71^save/Assign_72^save/Assign_73^save/Assign_74^save/Assign_75^save/Assign_76^save/Assign_77^save/Assign_78^save/Assign_79^save/Assign_8^save/Assign_80^save/Assign_81^save/Assign_82^save/Assign_83^save/Assign_84^save/Assign_85^save/Assign_9
-
save/restore_allNoOp^save/restore_shard
[
save_1/filename/inputConst*
dtype0*
_output_shapes
: *
valueB Bmodel
r
save_1/filenamePlaceholderWithDefaultsave_1/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_1/ConstPlaceholderWithDefaultsave_1/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_1/StringJoin/inputs_1Const*<
value3B1 B+_temp_6ec9cd7ec4ab43ebabc1ea03cd16b863/part*
dtype0*
_output_shapes
: 
{
save_1/StringJoin
StringJoinsave_1/Constsave_1/StringJoin/inputs_1*
_output_shapes
: *
	separator *
N
S
save_1/num_shardsConst*
dtype0*
_output_shapes
: *
value	B :
^
save_1/ShardedFilename/shardConst*
dtype0*
_output_shapes
: *
value	B : 
�
save_1/ShardedFilenameShardedFilenamesave_1/StringJoinsave_1/ShardedFilename/shardsave_1/num_shards*
_output_shapes
: 
�
save_1/SaveV2/tensor_namesConst*
dtype0*
_output_shapes
:V*�
value�B�VBRBR/AdamBR/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3B	log_alphaBlog_alpha/AdamBlog_alpha/Adam_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel
�
save_1/SaveV2/shape_and_slicesConst*�
value�B�VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:V
�
save_1/SaveV2SaveV2save_1/ShardedFilenamesave_1/SaveV2/tensor_namessave_1/SaveV2/shape_and_slicesRR/AdamR/Adam_1beta1_powerbeta1_power_1beta1_power_2beta1_power_3beta2_powerbeta2_power_1beta2_power_2beta2_power_3	log_alphalog_alpha/Adamlog_alpha/Adam_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1main/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1main/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1main/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1main/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1main/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1main/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1main/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1main/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1main/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1main/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1main/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q1/dense/biastarget/q1/dense/kerneltarget/q1/dense_1/biastarget/q1/dense_1/kerneltarget/q1/dense_2/biastarget/q1/dense_2/kerneltarget/q2/dense/biastarget/q2/dense/kerneltarget/q2/dense_1/biastarget/q2/dense_1/kerneltarget/q2/dense_2/biastarget/q2/dense_2/kernel*d
dtypesZ
X2V
�
save_1/control_dependencyIdentitysave_1/ShardedFilename^save_1/SaveV2*
T0*)
_class
loc:@save_1/ShardedFilename*
_output_shapes
: 
�
-save_1/MergeV2Checkpoints/checkpoint_prefixesPacksave_1/ShardedFilename^save_1/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_1/MergeV2CheckpointsMergeV2Checkpoints-save_1/MergeV2Checkpoints/checkpoint_prefixessave_1/Const*
delete_old_dirs(
�
save_1/IdentityIdentitysave_1/Const^save_1/MergeV2Checkpoints^save_1/control_dependency*
T0*
_output_shapes
: 
�
save_1/RestoreV2/tensor_namesConst*�
value�B�VBRBR/AdamBR/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3B	log_alphaBlog_alpha/AdamBlog_alpha/Adam_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel*
dtype0*
_output_shapes
:V
�
!save_1/RestoreV2/shape_and_slicesConst*�
value�B�VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:V
�
save_1/RestoreV2	RestoreV2save_1/Constsave_1/RestoreV2/tensor_names!save_1/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*d
dtypesZ
X2V
�
save_1/AssignAssignRsave_1/RestoreV2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class

loc:@R
�
save_1/Assign_1AssignR/Adamsave_1/RestoreV2:1*
_output_shapes

:*
use_locking(*
T0*
_class

loc:@R*
validate_shape(
�
save_1/Assign_2AssignR/Adam_1save_1/RestoreV2:2*
validate_shape(*
_output_shapes

:*
use_locking(*
T0*
_class

loc:@R
�
save_1/Assign_3Assignbeta1_powersave_1/RestoreV2:3*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_4Assignbeta1_power_1save_1/RestoreV2:4*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
�
save_1/Assign_5Assignbeta1_power_2save_1/RestoreV2:5*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias
�
save_1/Assign_6Assignbeta1_power_3save_1/RestoreV2:6*
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_7Assignbeta2_powersave_1/RestoreV2:7*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: 
�
save_1/Assign_8Assignbeta2_power_1save_1/RestoreV2:8*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_1/Assign_9Assignbeta2_power_2save_1/RestoreV2:9*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias
�
save_1/Assign_10Assignbeta2_power_3save_1/RestoreV2:10*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_1/Assign_11Assign	log_alphasave_1/RestoreV2:11*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha
�
save_1/Assign_12Assignlog_alpha/Adamsave_1/RestoreV2:12*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(
�
save_1/Assign_13Assignlog_alpha/Adam_1save_1/RestoreV2:13*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_1/Assign_14Assignmain/pi/dense/biassave_1/RestoreV2:14*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_15Assignmain/pi/dense/bias/Adamsave_1/RestoreV2:15*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_16Assignmain/pi/dense/bias/Adam_1save_1/RestoreV2:16*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_1/Assign_17Assignmain/pi/dense/kernelsave_1/RestoreV2:17*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_18Assignmain/pi/dense/kernel/Adamsave_1/RestoreV2:18*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel
�
save_1/Assign_19Assignmain/pi/dense/kernel/Adam_1save_1/RestoreV2:19*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_20Assignmain/pi/dense_1/biassave_1/RestoreV2:20*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_21Assignmain/pi/dense_1/bias/Adamsave_1/RestoreV2:21*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(
�
save_1/Assign_22Assignmain/pi/dense_1/bias/Adam_1save_1/RestoreV2:22*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_23Assignmain/pi/dense_1/kernelsave_1/RestoreV2:23*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel
�
save_1/Assign_24Assignmain/pi/dense_1/kernel/Adamsave_1/RestoreV2:24*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_25Assignmain/pi/dense_1/kernel/Adam_1save_1/RestoreV2:25*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_26Assignmain/pi/dense_2/biassave_1/RestoreV2:26*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_27Assignmain/pi/dense_2/bias/Adamsave_1/RestoreV2:27*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(
�
save_1/Assign_28Assignmain/pi/dense_2/bias/Adam_1save_1/RestoreV2:28*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_29Assignmain/pi/dense_2/kernelsave_1/RestoreV2:29*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_1/Assign_30Assignmain/pi/dense_2/kernel/Adamsave_1/RestoreV2:30*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(
�
save_1/Assign_31Assignmain/pi/dense_2/kernel/Adam_1save_1/RestoreV2:31*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_1/Assign_32Assignmain/q1/dense/biassave_1/RestoreV2:32*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_33Assignmain/q1/dense/bias/Adamsave_1/RestoreV2:33*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_1/Assign_34Assignmain/q1/dense/bias/Adam_1save_1/RestoreV2:34*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_35Assignmain/q1/dense/kernelsave_1/RestoreV2:35*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_1/Assign_36Assignmain/q1/dense/kernel/Adamsave_1/RestoreV2:36*
_output_shapes

:@*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(
�
save_1/Assign_37Assignmain/q1/dense/kernel/Adam_1save_1/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(
�
save_1/Assign_38Assignmain/q1/dense_1/biassave_1/RestoreV2:38*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_39Assignmain/q1/dense_1/bias/Adamsave_1/RestoreV2:39*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_1/Assign_40Assignmain/q1/dense_1/bias/Adam_1save_1/RestoreV2:40*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_41Assignmain/q1/dense_1/kernelsave_1/RestoreV2:41*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_42Assignmain/q1/dense_1/kernel/Adamsave_1/RestoreV2:42*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_1/Assign_43Assignmain/q1/dense_1/kernel/Adam_1save_1/RestoreV2:43*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_44Assignmain/q1/dense_2/biassave_1/RestoreV2:44*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_45Assignmain/q1/dense_2/bias/Adamsave_1/RestoreV2:45*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(
�
save_1/Assign_46Assignmain/q1/dense_2/bias/Adam_1save_1/RestoreV2:46*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_1/Assign_47Assignmain/q1/dense_2/kernelsave_1/RestoreV2:47*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_1/Assign_48Assignmain/q1/dense_2/kernel/Adamsave_1/RestoreV2:48*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(
�
save_1/Assign_49Assignmain/q1/dense_2/kernel/Adam_1save_1/RestoreV2:49*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_1/Assign_50Assignmain/q2/dense/biassave_1/RestoreV2:50*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_51Assignmain/q2/dense/bias/Adamsave_1/RestoreV2:51*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(
�
save_1/Assign_52Assignmain/q2/dense/bias/Adam_1save_1/RestoreV2:52*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_53Assignmain/q2/dense/kernelsave_1/RestoreV2:53*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_54Assignmain/q2/dense/kernel/Adamsave_1/RestoreV2:54*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel
�
save_1/Assign_55Assignmain/q2/dense/kernel/Adam_1save_1/RestoreV2:55*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_1/Assign_56Assignmain/q2/dense_1/biassave_1/RestoreV2:56*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(
�
save_1/Assign_57Assignmain/q2/dense_1/bias/Adamsave_1/RestoreV2:57*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(
�
save_1/Assign_58Assignmain/q2/dense_1/bias/Adam_1save_1/RestoreV2:58*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_59Assignmain/q2/dense_1/kernelsave_1/RestoreV2:59*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel
�
save_1/Assign_60Assignmain/q2/dense_1/kernel/Adamsave_1/RestoreV2:60*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_1/Assign_61Assignmain/q2/dense_1/kernel/Adam_1save_1/RestoreV2:61*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_62Assignmain/q2/dense_2/biassave_1/RestoreV2:62*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_63Assignmain/q2/dense_2/bias/Adamsave_1/RestoreV2:63*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_64Assignmain/q2/dense_2/bias/Adam_1save_1/RestoreV2:64*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_1/Assign_65Assignmain/q2/dense_2/kernelsave_1/RestoreV2:65*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_66Assignmain/q2/dense_2/kernel/Adamsave_1/RestoreV2:66*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_67Assignmain/q2/dense_2/kernel/Adam_1save_1/RestoreV2:67*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_68Assigntarget/pi/dense/biassave_1/RestoreV2:68*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_1/Assign_69Assigntarget/pi/dense/kernelsave_1/RestoreV2:69*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_70Assigntarget/pi/dense_1/biassave_1/RestoreV2:70*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias
�
save_1/Assign_71Assigntarget/pi/dense_1/kernelsave_1/RestoreV2:71*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_1/Assign_72Assigntarget/pi/dense_2/biassave_1/RestoreV2:72*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_73Assigntarget/pi/dense_2/kernelsave_1/RestoreV2:73*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_74Assigntarget/q1/dense/biassave_1/RestoreV2:74*
T0*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_75Assigntarget/q1/dense/kernelsave_1/RestoreV2:75*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@target/q1/dense/kernel
�
save_1/Assign_76Assigntarget/q1/dense_1/biassave_1/RestoreV2:76*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_77Assigntarget/q1/dense_1/kernelsave_1/RestoreV2:77*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_1/Assign_78Assigntarget/q1/dense_2/biassave_1/RestoreV2:78*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_1/Assign_79Assigntarget/q1/dense_2/kernelsave_1/RestoreV2:79*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_1/Assign_80Assigntarget/q2/dense/biassave_1/RestoreV2:80*
T0*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_81Assigntarget/q2/dense/kernelsave_1/RestoreV2:81*
use_locking(*
T0*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_1/Assign_82Assigntarget/q2/dense_1/biassave_1/RestoreV2:82*
T0*)
_class
loc:@target/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_1/Assign_83Assigntarget/q2/dense_1/kernelsave_1/RestoreV2:83*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_1/Assign_84Assigntarget/q2/dense_2/biassave_1/RestoreV2:84*
use_locking(*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_1/Assign_85Assigntarget/q2/dense_2/kernelsave_1/RestoreV2:85*
_output_shapes

:@*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(
�
save_1/restore_shardNoOp^save_1/Assign^save_1/Assign_1^save_1/Assign_10^save_1/Assign_11^save_1/Assign_12^save_1/Assign_13^save_1/Assign_14^save_1/Assign_15^save_1/Assign_16^save_1/Assign_17^save_1/Assign_18^save_1/Assign_19^save_1/Assign_2^save_1/Assign_20^save_1/Assign_21^save_1/Assign_22^save_1/Assign_23^save_1/Assign_24^save_1/Assign_25^save_1/Assign_26^save_1/Assign_27^save_1/Assign_28^save_1/Assign_29^save_1/Assign_3^save_1/Assign_30^save_1/Assign_31^save_1/Assign_32^save_1/Assign_33^save_1/Assign_34^save_1/Assign_35^save_1/Assign_36^save_1/Assign_37^save_1/Assign_38^save_1/Assign_39^save_1/Assign_4^save_1/Assign_40^save_1/Assign_41^save_1/Assign_42^save_1/Assign_43^save_1/Assign_44^save_1/Assign_45^save_1/Assign_46^save_1/Assign_47^save_1/Assign_48^save_1/Assign_49^save_1/Assign_5^save_1/Assign_50^save_1/Assign_51^save_1/Assign_52^save_1/Assign_53^save_1/Assign_54^save_1/Assign_55^save_1/Assign_56^save_1/Assign_57^save_1/Assign_58^save_1/Assign_59^save_1/Assign_6^save_1/Assign_60^save_1/Assign_61^save_1/Assign_62^save_1/Assign_63^save_1/Assign_64^save_1/Assign_65^save_1/Assign_66^save_1/Assign_67^save_1/Assign_68^save_1/Assign_69^save_1/Assign_7^save_1/Assign_70^save_1/Assign_71^save_1/Assign_72^save_1/Assign_73^save_1/Assign_74^save_1/Assign_75^save_1/Assign_76^save_1/Assign_77^save_1/Assign_78^save_1/Assign_79^save_1/Assign_8^save_1/Assign_80^save_1/Assign_81^save_1/Assign_82^save_1/Assign_83^save_1/Assign_84^save_1/Assign_85^save_1/Assign_9
1
save_1/restore_allNoOp^save_1/restore_shard
[
save_2/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_2/filenamePlaceholderWithDefaultsave_2/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_2/ConstPlaceholderWithDefaultsave_2/filename*
dtype0*
_output_shapes
: *
shape: 
�
save_2/StringJoin/inputs_1Const*
dtype0*
_output_shapes
: *<
value3B1 B+_temp_943a9dbf39684251b6fe3aabf5092f3e/part
{
save_2/StringJoin
StringJoinsave_2/Constsave_2/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_2/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_2/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_2/ShardedFilenameShardedFilenamesave_2/StringJoinsave_2/ShardedFilename/shardsave_2/num_shards*
_output_shapes
: 
�
save_2/SaveV2/tensor_namesConst*�
value�B�VBRBR/AdamBR/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3B	log_alphaBlog_alpha/AdamBlog_alpha/Adam_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel*
dtype0*
_output_shapes
:V
�
save_2/SaveV2/shape_and_slicesConst*�
value�B�VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:V
�
save_2/SaveV2SaveV2save_2/ShardedFilenamesave_2/SaveV2/tensor_namessave_2/SaveV2/shape_and_slicesRR/AdamR/Adam_1beta1_powerbeta1_power_1beta1_power_2beta1_power_3beta2_powerbeta2_power_1beta2_power_2beta2_power_3	log_alphalog_alpha/Adamlog_alpha/Adam_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1main/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1main/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1main/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1main/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1main/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1main/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1main/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1main/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1main/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1main/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1main/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q1/dense/biastarget/q1/dense/kerneltarget/q1/dense_1/biastarget/q1/dense_1/kerneltarget/q1/dense_2/biastarget/q1/dense_2/kerneltarget/q2/dense/biastarget/q2/dense/kerneltarget/q2/dense_1/biastarget/q2/dense_1/kerneltarget/q2/dense_2/biastarget/q2/dense_2/kernel*d
dtypesZ
X2V
�
save_2/control_dependencyIdentitysave_2/ShardedFilename^save_2/SaveV2*
T0*)
_class
loc:@save_2/ShardedFilename*
_output_shapes
: 
�
-save_2/MergeV2Checkpoints/checkpoint_prefixesPacksave_2/ShardedFilename^save_2/control_dependency*
N*
_output_shapes
:*
T0*

axis 
�
save_2/MergeV2CheckpointsMergeV2Checkpoints-save_2/MergeV2Checkpoints/checkpoint_prefixessave_2/Const*
delete_old_dirs(
�
save_2/IdentityIdentitysave_2/Const^save_2/MergeV2Checkpoints^save_2/control_dependency*
T0*
_output_shapes
: 
�
save_2/RestoreV2/tensor_namesConst*
dtype0*
_output_shapes
:V*�
value�B�VBRBR/AdamBR/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3B	log_alphaBlog_alpha/AdamBlog_alpha/Adam_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel
�
!save_2/RestoreV2/shape_and_slicesConst*�
value�B�VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:V
�
save_2/RestoreV2	RestoreV2save_2/Constsave_2/RestoreV2/tensor_names!save_2/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*d
dtypesZ
X2V
�
save_2/AssignAssignRsave_2/RestoreV2*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes

:
�
save_2/Assign_1AssignR/Adamsave_2/RestoreV2:1*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes

:
�
save_2/Assign_2AssignR/Adam_1save_2/RestoreV2:2*
_output_shapes

:*
use_locking(*
T0*
_class

loc:@R*
validate_shape(
�
save_2/Assign_3Assignbeta1_powersave_2/RestoreV2:3*
_class

loc:@R*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_2/Assign_4Assignbeta1_power_1save_2/RestoreV2:4*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
�
save_2/Assign_5Assignbeta1_power_2save_2/RestoreV2:5*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_6Assignbeta1_power_3save_2/RestoreV2:6*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha
�
save_2/Assign_7Assignbeta2_powersave_2/RestoreV2:7*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_8Assignbeta2_power_1save_2/RestoreV2:8*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_2/Assign_9Assignbeta2_power_2save_2/RestoreV2:9*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_10Assignbeta2_power_3save_2/RestoreV2:10*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha
�
save_2/Assign_11Assign	log_alphasave_2/RestoreV2:11*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha
�
save_2/Assign_12Assignlog_alpha/Adamsave_2/RestoreV2:12*
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_13Assignlog_alpha/Adam_1save_2/RestoreV2:13*
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: 
�
save_2/Assign_14Assignmain/pi/dense/biassave_2/RestoreV2:14*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
�
save_2/Assign_15Assignmain/pi/dense/bias/Adamsave_2/RestoreV2:15*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_16Assignmain/pi/dense/bias/Adam_1save_2/RestoreV2:16*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(
�
save_2/Assign_17Assignmain/pi/dense/kernelsave_2/RestoreV2:17*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_18Assignmain/pi/dense/kernel/Adamsave_2/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_19Assignmain/pi/dense/kernel/Adam_1save_2/RestoreV2:19*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_2/Assign_20Assignmain/pi/dense_1/biassave_2/RestoreV2:20*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias
�
save_2/Assign_21Assignmain/pi/dense_1/bias/Adamsave_2/RestoreV2:21*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_2/Assign_22Assignmain/pi/dense_1/bias/Adam_1save_2/RestoreV2:22*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_23Assignmain/pi/dense_1/kernelsave_2/RestoreV2:23*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_24Assignmain/pi/dense_1/kernel/Adamsave_2/RestoreV2:24*
_output_shapes

:@@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(
�
save_2/Assign_25Assignmain/pi/dense_1/kernel/Adam_1save_2/RestoreV2:25*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_2/Assign_26Assignmain/pi/dense_2/biassave_2/RestoreV2:26*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_2/Assign_27Assignmain/pi/dense_2/bias/Adamsave_2/RestoreV2:27*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_28Assignmain/pi/dense_2/bias/Adam_1save_2/RestoreV2:28*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(
�
save_2/Assign_29Assignmain/pi/dense_2/kernelsave_2/RestoreV2:29*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_30Assignmain/pi/dense_2/kernel/Adamsave_2/RestoreV2:30*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_31Assignmain/pi/dense_2/kernel/Adam_1save_2/RestoreV2:31*
T0*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_2/Assign_32Assignmain/q1/dense/biassave_2/RestoreV2:32*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_2/Assign_33Assignmain/q1/dense/bias/Adamsave_2/RestoreV2:33*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias
�
save_2/Assign_34Assignmain/q1/dense/bias/Adam_1save_2/RestoreV2:34*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(
�
save_2/Assign_35Assignmain/q1/dense/kernelsave_2/RestoreV2:35*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_36Assignmain/q1/dense/kernel/Adamsave_2/RestoreV2:36*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_37Assignmain/q1/dense/kernel/Adam_1save_2/RestoreV2:37*
_output_shapes

:@*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(
�
save_2/Assign_38Assignmain/q1/dense_1/biassave_2/RestoreV2:38*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_2/Assign_39Assignmain/q1/dense_1/bias/Adamsave_2/RestoreV2:39*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_40Assignmain/q1/dense_1/bias/Adam_1save_2/RestoreV2:40*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_2/Assign_41Assignmain/q1/dense_1/kernelsave_2/RestoreV2:41*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_2/Assign_42Assignmain/q1/dense_1/kernel/Adamsave_2/RestoreV2:42*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_2/Assign_43Assignmain/q1/dense_1/kernel/Adam_1save_2/RestoreV2:43*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_2/Assign_44Assignmain/q1/dense_2/biassave_2/RestoreV2:44*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_45Assignmain/q1/dense_2/bias/Adamsave_2/RestoreV2:45*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_2/Assign_46Assignmain/q1/dense_2/bias/Adam_1save_2/RestoreV2:46*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_2/Assign_47Assignmain/q1/dense_2/kernelsave_2/RestoreV2:47*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_2/Assign_48Assignmain/q1/dense_2/kernel/Adamsave_2/RestoreV2:48*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_49Assignmain/q1/dense_2/kernel/Adam_1save_2/RestoreV2:49*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_50Assignmain/q2/dense/biassave_2/RestoreV2:50*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias
�
save_2/Assign_51Assignmain/q2/dense/bias/Adamsave_2/RestoreV2:51*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_52Assignmain/q2/dense/bias/Adam_1save_2/RestoreV2:52*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_2/Assign_53Assignmain/q2/dense/kernelsave_2/RestoreV2:53*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_54Assignmain/q2/dense/kernel/Adamsave_2/RestoreV2:54*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_55Assignmain/q2/dense/kernel/Adam_1save_2/RestoreV2:55*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_2/Assign_56Assignmain/q2/dense_1/biassave_2/RestoreV2:56*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_57Assignmain/q2/dense_1/bias/Adamsave_2/RestoreV2:57*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias
�
save_2/Assign_58Assignmain/q2/dense_1/bias/Adam_1save_2/RestoreV2:58*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_59Assignmain/q2/dense_1/kernelsave_2/RestoreV2:59*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_2/Assign_60Assignmain/q2/dense_1/kernel/Adamsave_2/RestoreV2:60*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_2/Assign_61Assignmain/q2/dense_1/kernel/Adam_1save_2/RestoreV2:61*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_2/Assign_62Assignmain/q2/dense_2/biassave_2/RestoreV2:62*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_63Assignmain/q2/dense_2/bias/Adamsave_2/RestoreV2:63*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_2/Assign_64Assignmain/q2/dense_2/bias/Adam_1save_2/RestoreV2:64*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias
�
save_2/Assign_65Assignmain/q2/dense_2/kernelsave_2/RestoreV2:65*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(
�
save_2/Assign_66Assignmain/q2/dense_2/kernel/Adamsave_2/RestoreV2:66*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel
�
save_2/Assign_67Assignmain/q2/dense_2/kernel/Adam_1save_2/RestoreV2:67*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_68Assigntarget/pi/dense/biassave_2/RestoreV2:68*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias
�
save_2/Assign_69Assigntarget/pi/dense/kernelsave_2/RestoreV2:69*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_70Assigntarget/pi/dense_1/biassave_2/RestoreV2:70*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_71Assigntarget/pi/dense_1/kernelsave_2/RestoreV2:71*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_72Assigntarget/pi/dense_2/biassave_2/RestoreV2:72*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_2/Assign_73Assigntarget/pi/dense_2/kernelsave_2/RestoreV2:73*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel
�
save_2/Assign_74Assigntarget/q1/dense/biassave_2/RestoreV2:74*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*'
_class
loc:@target/q1/dense/bias
�
save_2/Assign_75Assigntarget/q1/dense/kernelsave_2/RestoreV2:75*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@target/q1/dense/kernel*
validate_shape(
�
save_2/Assign_76Assigntarget/q1/dense_1/biassave_2/RestoreV2:76*
use_locking(*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_2/Assign_77Assigntarget/q1/dense_1/kernelsave_2/RestoreV2:77*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_78Assigntarget/q1/dense_2/biassave_2/RestoreV2:78*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(*
T0
�
save_2/Assign_79Assigntarget/q1/dense_2/kernelsave_2/RestoreV2:79*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_80Assigntarget/q2/dense/biassave_2/RestoreV2:80*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_2/Assign_81Assigntarget/q2/dense/kernelsave_2/RestoreV2:81*
use_locking(*
T0*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/Assign_82Assigntarget/q2/dense_1/biassave_2/RestoreV2:82*
T0*)
_class
loc:@target/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_2/Assign_83Assigntarget/q2/dense_1/kernelsave_2/RestoreV2:83*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_2/Assign_84Assigntarget/q2/dense_2/biassave_2/RestoreV2:84*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_2/Assign_85Assigntarget/q2/dense_2/kernelsave_2/RestoreV2:85*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_2/restore_shardNoOp^save_2/Assign^save_2/Assign_1^save_2/Assign_10^save_2/Assign_11^save_2/Assign_12^save_2/Assign_13^save_2/Assign_14^save_2/Assign_15^save_2/Assign_16^save_2/Assign_17^save_2/Assign_18^save_2/Assign_19^save_2/Assign_2^save_2/Assign_20^save_2/Assign_21^save_2/Assign_22^save_2/Assign_23^save_2/Assign_24^save_2/Assign_25^save_2/Assign_26^save_2/Assign_27^save_2/Assign_28^save_2/Assign_29^save_2/Assign_3^save_2/Assign_30^save_2/Assign_31^save_2/Assign_32^save_2/Assign_33^save_2/Assign_34^save_2/Assign_35^save_2/Assign_36^save_2/Assign_37^save_2/Assign_38^save_2/Assign_39^save_2/Assign_4^save_2/Assign_40^save_2/Assign_41^save_2/Assign_42^save_2/Assign_43^save_2/Assign_44^save_2/Assign_45^save_2/Assign_46^save_2/Assign_47^save_2/Assign_48^save_2/Assign_49^save_2/Assign_5^save_2/Assign_50^save_2/Assign_51^save_2/Assign_52^save_2/Assign_53^save_2/Assign_54^save_2/Assign_55^save_2/Assign_56^save_2/Assign_57^save_2/Assign_58^save_2/Assign_59^save_2/Assign_6^save_2/Assign_60^save_2/Assign_61^save_2/Assign_62^save_2/Assign_63^save_2/Assign_64^save_2/Assign_65^save_2/Assign_66^save_2/Assign_67^save_2/Assign_68^save_2/Assign_69^save_2/Assign_7^save_2/Assign_70^save_2/Assign_71^save_2/Assign_72^save_2/Assign_73^save_2/Assign_74^save_2/Assign_75^save_2/Assign_76^save_2/Assign_77^save_2/Assign_78^save_2/Assign_79^save_2/Assign_8^save_2/Assign_80^save_2/Assign_81^save_2/Assign_82^save_2/Assign_83^save_2/Assign_84^save_2/Assign_85^save_2/Assign_9
1
save_2/restore_allNoOp^save_2/restore_shard
[
save_3/filename/inputConst*
valueB Bmodel*
dtype0*
_output_shapes
: 
r
save_3/filenamePlaceholderWithDefaultsave_3/filename/input*
dtype0*
_output_shapes
: *
shape: 
i
save_3/ConstPlaceholderWithDefaultsave_3/filename*
shape: *
dtype0*
_output_shapes
: 
�
save_3/StringJoin/inputs_1Const*<
value3B1 B+_temp_876106dc0c7848588fc721b8457e0ab7/part*
dtype0*
_output_shapes
: 
{
save_3/StringJoin
StringJoinsave_3/Constsave_3/StringJoin/inputs_1*
	separator *
N*
_output_shapes
: 
S
save_3/num_shardsConst*
value	B :*
dtype0*
_output_shapes
: 
^
save_3/ShardedFilename/shardConst*
value	B : *
dtype0*
_output_shapes
: 
�
save_3/ShardedFilenameShardedFilenamesave_3/StringJoinsave_3/ShardedFilename/shardsave_3/num_shards*
_output_shapes
: 
�
save_3/SaveV2/tensor_namesConst*�
value�B�VBRBR/AdamBR/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3B	log_alphaBlog_alpha/AdamBlog_alpha/Adam_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel*
dtype0*
_output_shapes
:V
�
save_3/SaveV2/shape_and_slicesConst*�
value�B�VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:V
�
save_3/SaveV2SaveV2save_3/ShardedFilenamesave_3/SaveV2/tensor_namessave_3/SaveV2/shape_and_slicesRR/AdamR/Adam_1beta1_powerbeta1_power_1beta1_power_2beta1_power_3beta2_powerbeta2_power_1beta2_power_2beta2_power_3	log_alphalog_alpha/Adamlog_alpha/Adam_1main/pi/dense/biasmain/pi/dense/bias/Adammain/pi/dense/bias/Adam_1main/pi/dense/kernelmain/pi/dense/kernel/Adammain/pi/dense/kernel/Adam_1main/pi/dense_1/biasmain/pi/dense_1/bias/Adammain/pi/dense_1/bias/Adam_1main/pi/dense_1/kernelmain/pi/dense_1/kernel/Adammain/pi/dense_1/kernel/Adam_1main/pi/dense_2/biasmain/pi/dense_2/bias/Adammain/pi/dense_2/bias/Adam_1main/pi/dense_2/kernelmain/pi/dense_2/kernel/Adammain/pi/dense_2/kernel/Adam_1main/q1/dense/biasmain/q1/dense/bias/Adammain/q1/dense/bias/Adam_1main/q1/dense/kernelmain/q1/dense/kernel/Adammain/q1/dense/kernel/Adam_1main/q1/dense_1/biasmain/q1/dense_1/bias/Adammain/q1/dense_1/bias/Adam_1main/q1/dense_1/kernelmain/q1/dense_1/kernel/Adammain/q1/dense_1/kernel/Adam_1main/q1/dense_2/biasmain/q1/dense_2/bias/Adammain/q1/dense_2/bias/Adam_1main/q1/dense_2/kernelmain/q1/dense_2/kernel/Adammain/q1/dense_2/kernel/Adam_1main/q2/dense/biasmain/q2/dense/bias/Adammain/q2/dense/bias/Adam_1main/q2/dense/kernelmain/q2/dense/kernel/Adammain/q2/dense/kernel/Adam_1main/q2/dense_1/biasmain/q2/dense_1/bias/Adammain/q2/dense_1/bias/Adam_1main/q2/dense_1/kernelmain/q2/dense_1/kernel/Adammain/q2/dense_1/kernel/Adam_1main/q2/dense_2/biasmain/q2/dense_2/bias/Adammain/q2/dense_2/bias/Adam_1main/q2/dense_2/kernelmain/q2/dense_2/kernel/Adammain/q2/dense_2/kernel/Adam_1target/pi/dense/biastarget/pi/dense/kerneltarget/pi/dense_1/biastarget/pi/dense_1/kerneltarget/pi/dense_2/biastarget/pi/dense_2/kerneltarget/q1/dense/biastarget/q1/dense/kerneltarget/q1/dense_1/biastarget/q1/dense_1/kerneltarget/q1/dense_2/biastarget/q1/dense_2/kerneltarget/q2/dense/biastarget/q2/dense/kerneltarget/q2/dense_1/biastarget/q2/dense_1/kerneltarget/q2/dense_2/biastarget/q2/dense_2/kernel*d
dtypesZ
X2V
�
save_3/control_dependencyIdentitysave_3/ShardedFilename^save_3/SaveV2*
T0*)
_class
loc:@save_3/ShardedFilename*
_output_shapes
: 
�
-save_3/MergeV2Checkpoints/checkpoint_prefixesPacksave_3/ShardedFilename^save_3/control_dependency*
T0*

axis *
N*
_output_shapes
:
�
save_3/MergeV2CheckpointsMergeV2Checkpoints-save_3/MergeV2Checkpoints/checkpoint_prefixessave_3/Const*
delete_old_dirs(
�
save_3/IdentityIdentitysave_3/Const^save_3/MergeV2Checkpoints^save_3/control_dependency*
_output_shapes
: *
T0
�
save_3/RestoreV2/tensor_namesConst*�
value�B�VBRBR/AdamBR/Adam_1Bbeta1_powerBbeta1_power_1Bbeta1_power_2Bbeta1_power_3Bbeta2_powerBbeta2_power_1Bbeta2_power_2Bbeta2_power_3B	log_alphaBlog_alpha/AdamBlog_alpha/Adam_1Bmain/pi/dense/biasBmain/pi/dense/bias/AdamBmain/pi/dense/bias/Adam_1Bmain/pi/dense/kernelBmain/pi/dense/kernel/AdamBmain/pi/dense/kernel/Adam_1Bmain/pi/dense_1/biasBmain/pi/dense_1/bias/AdamBmain/pi/dense_1/bias/Adam_1Bmain/pi/dense_1/kernelBmain/pi/dense_1/kernel/AdamBmain/pi/dense_1/kernel/Adam_1Bmain/pi/dense_2/biasBmain/pi/dense_2/bias/AdamBmain/pi/dense_2/bias/Adam_1Bmain/pi/dense_2/kernelBmain/pi/dense_2/kernel/AdamBmain/pi/dense_2/kernel/Adam_1Bmain/q1/dense/biasBmain/q1/dense/bias/AdamBmain/q1/dense/bias/Adam_1Bmain/q1/dense/kernelBmain/q1/dense/kernel/AdamBmain/q1/dense/kernel/Adam_1Bmain/q1/dense_1/biasBmain/q1/dense_1/bias/AdamBmain/q1/dense_1/bias/Adam_1Bmain/q1/dense_1/kernelBmain/q1/dense_1/kernel/AdamBmain/q1/dense_1/kernel/Adam_1Bmain/q1/dense_2/biasBmain/q1/dense_2/bias/AdamBmain/q1/dense_2/bias/Adam_1Bmain/q1/dense_2/kernelBmain/q1/dense_2/kernel/AdamBmain/q1/dense_2/kernel/Adam_1Bmain/q2/dense/biasBmain/q2/dense/bias/AdamBmain/q2/dense/bias/Adam_1Bmain/q2/dense/kernelBmain/q2/dense/kernel/AdamBmain/q2/dense/kernel/Adam_1Bmain/q2/dense_1/biasBmain/q2/dense_1/bias/AdamBmain/q2/dense_1/bias/Adam_1Bmain/q2/dense_1/kernelBmain/q2/dense_1/kernel/AdamBmain/q2/dense_1/kernel/Adam_1Bmain/q2/dense_2/biasBmain/q2/dense_2/bias/AdamBmain/q2/dense_2/bias/Adam_1Bmain/q2/dense_2/kernelBmain/q2/dense_2/kernel/AdamBmain/q2/dense_2/kernel/Adam_1Btarget/pi/dense/biasBtarget/pi/dense/kernelBtarget/pi/dense_1/biasBtarget/pi/dense_1/kernelBtarget/pi/dense_2/biasBtarget/pi/dense_2/kernelBtarget/q1/dense/biasBtarget/q1/dense/kernelBtarget/q1/dense_1/biasBtarget/q1/dense_1/kernelBtarget/q1/dense_2/biasBtarget/q1/dense_2/kernelBtarget/q2/dense/biasBtarget/q2/dense/kernelBtarget/q2/dense_1/biasBtarget/q2/dense_1/kernelBtarget/q2/dense_2/biasBtarget/q2/dense_2/kernel*
dtype0*
_output_shapes
:V
�
!save_3/RestoreV2/shape_and_slicesConst*�
value�B�VB B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B B *
dtype0*
_output_shapes
:V
�
save_3/RestoreV2	RestoreV2save_3/Constsave_3/RestoreV2/tensor_names!save_3/RestoreV2/shape_and_slices*�
_output_shapes�
�::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::::*d
dtypesZ
X2V
�
save_3/AssignAssignRsave_3/RestoreV2*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes

:
�
save_3/Assign_1AssignR/Adamsave_3/RestoreV2:1*
_output_shapes

:*
use_locking(*
T0*
_class

loc:@R*
validate_shape(
�
save_3/Assign_2AssignR/Adam_1save_3/RestoreV2:2*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes

:
�
save_3/Assign_3Assignbeta1_powersave_3/RestoreV2:3*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class

loc:@R
�
save_3/Assign_4Assignbeta1_power_1save_3/RestoreV2:4*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_3/Assign_5Assignbeta1_power_2save_3/RestoreV2:5*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(*
T0
�
save_3/Assign_6Assignbeta1_power_3save_3/RestoreV2:6*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_7Assignbeta2_powersave_3/RestoreV2:7*
use_locking(*
T0*
_class

loc:@R*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_8Assignbeta2_power_1save_3/RestoreV2:8*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_9Assignbeta2_power_2save_3/RestoreV2:9*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_10Assignbeta2_power_3save_3/RestoreV2:10*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_11Assign	log_alphasave_3/RestoreV2:11*
use_locking(*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: 
�
save_3/Assign_12Assignlog_alpha/Adamsave_3/RestoreV2:12*
T0*
_class
loc:@log_alpha*
validate_shape(*
_output_shapes
: *
use_locking(
�
save_3/Assign_13Assignlog_alpha/Adam_1save_3/RestoreV2:13*
validate_shape(*
_output_shapes
: *
use_locking(*
T0*
_class
loc:@log_alpha
�
save_3/Assign_14Assignmain/pi/dense/biassave_3/RestoreV2:14*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias
�
save_3/Assign_15Assignmain/pi/dense/bias/Adamsave_3/RestoreV2:15*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_16Assignmain/pi/dense/bias/Adam_1save_3/RestoreV2:16*
use_locking(*
T0*%
_class
loc:@main/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_17Assignmain/pi/dense/kernelsave_3/RestoreV2:17*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_3/Assign_18Assignmain/pi/dense/kernel/Adamsave_3/RestoreV2:18*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_19Assignmain/pi/dense/kernel/Adam_1save_3/RestoreV2:19*
use_locking(*
T0*'
_class
loc:@main/pi/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_20Assignmain/pi/dense_1/biassave_3/RestoreV2:20*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_21Assignmain/pi/dense_1/bias/Adamsave_3/RestoreV2:21*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_3/Assign_22Assignmain/pi/dense_1/bias/Adam_1save_3/RestoreV2:22*
use_locking(*
T0*'
_class
loc:@main/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_23Assignmain/pi/dense_1/kernelsave_3/RestoreV2:23*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(
�
save_3/Assign_24Assignmain/pi/dense_1/kernel/Adamsave_3/RestoreV2:24*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_25Assignmain/pi/dense_1/kernel/Adam_1save_3/RestoreV2:25*
use_locking(*
T0*)
_class
loc:@main/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_26Assignmain/pi/dense_2/biassave_3/RestoreV2:26*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_27Assignmain/pi/dense_2/bias/Adamsave_3/RestoreV2:27*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_28Assignmain/pi/dense_2/bias/Adam_1save_3/RestoreV2:28*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/pi/dense_2/bias
�
save_3/Assign_29Assignmain/pi/dense_2/kernelsave_3/RestoreV2:29*)
_class
loc:@main/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_3/Assign_30Assignmain/pi/dense_2/kernel/Adamsave_3/RestoreV2:30*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel
�
save_3/Assign_31Assignmain/pi/dense_2/kernel/Adam_1save_3/RestoreV2:31*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/pi/dense_2/kernel
�
save_3/Assign_32Assignmain/q1/dense/biassave_3/RestoreV2:32*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_3/Assign_33Assignmain/q1/dense/bias/Adamsave_3/RestoreV2:33*
use_locking(*
T0*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_34Assignmain/q1/dense/bias/Adam_1save_3/RestoreV2:34*%
_class
loc:@main/q1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_3/Assign_35Assignmain/q1/dense/kernelsave_3/RestoreV2:35*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_36Assignmain/q1/dense/kernel/Adamsave_3/RestoreV2:36*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_37Assignmain/q1/dense/kernel/Adam_1save_3/RestoreV2:37*
use_locking(*
T0*'
_class
loc:@main/q1/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_38Assignmain/q1/dense_1/biassave_3/RestoreV2:38*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_39Assignmain/q1/dense_1/bias/Adamsave_3/RestoreV2:39*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_3/Assign_40Assignmain/q1/dense_1/bias/Adam_1save_3/RestoreV2:40*
use_locking(*
T0*'
_class
loc:@main/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_41Assignmain/q1/dense_1/kernelsave_3/RestoreV2:41*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_42Assignmain/q1/dense_1/kernel/Adamsave_3/RestoreV2:42*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_43Assignmain/q1/dense_1/kernel/Adam_1save_3/RestoreV2:43*
use_locking(*
T0*)
_class
loc:@main/q1/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_44Assignmain/q1/dense_2/biassave_3/RestoreV2:44*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias
�
save_3/Assign_45Assignmain/q1/dense_2/bias/Adamsave_3/RestoreV2:45*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_46Assignmain/q1/dense_2/bias/Adam_1save_3/RestoreV2:46*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q1/dense_2/bias*
validate_shape(
�
save_3/Assign_47Assignmain/q1/dense_2/kernelsave_3/RestoreV2:47*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel
�
save_3/Assign_48Assignmain/q1/dense_2/kernel/Adamsave_3/RestoreV2:48*)
_class
loc:@main/q1/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_3/Assign_49Assignmain/q1/dense_2/kernel/Adam_1save_3/RestoreV2:49*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q1/dense_2/kernel
�
save_3/Assign_50Assignmain/q2/dense/biassave_3/RestoreV2:50*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_51Assignmain/q2/dense/bias/Adamsave_3/RestoreV2:51*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias
�
save_3/Assign_52Assignmain/q2/dense/bias/Adam_1save_3/RestoreV2:52*
use_locking(*
T0*%
_class
loc:@main/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_53Assignmain/q2/dense/kernelsave_3/RestoreV2:53*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_54Assignmain/q2/dense/kernel/Adamsave_3/RestoreV2:54*
_output_shapes

:@*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(
�
save_3/Assign_55Assignmain/q2/dense/kernel/Adam_1save_3/RestoreV2:55*
_output_shapes

:@*
use_locking(*
T0*'
_class
loc:@main/q2/dense/kernel*
validate_shape(
�
save_3/Assign_56Assignmain/q2/dense_1/biassave_3/RestoreV2:56*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_57Assignmain/q2/dense_1/bias/Adamsave_3/RestoreV2:57*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_3/Assign_58Assignmain/q2/dense_1/bias/Adam_1save_3/RestoreV2:58*
use_locking(*
T0*'
_class
loc:@main/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_59Assignmain/q2/dense_1/kernelsave_3/RestoreV2:59*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_60Assignmain/q2/dense_1/kernel/Adamsave_3/RestoreV2:60*
_output_shapes

:@@*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel*
validate_shape(
�
save_3/Assign_61Assignmain/q2/dense_1/kernel/Adam_1save_3/RestoreV2:61*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0*)
_class
loc:@main/q2/dense_1/kernel
�
save_3/Assign_62Assignmain/q2/dense_2/biassave_3/RestoreV2:62*
validate_shape(*
_output_shapes
:*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias
�
save_3/Assign_63Assignmain/q2/dense_2/bias/Adamsave_3/RestoreV2:63*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_3/Assign_64Assignmain/q2/dense_2/bias/Adam_1save_3/RestoreV2:64*
use_locking(*
T0*'
_class
loc:@main/q2/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_65Assignmain/q2/dense_2/kernelsave_3/RestoreV2:65*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_66Assignmain/q2/dense_2/kernel/Adamsave_3/RestoreV2:66*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0
�
save_3/Assign_67Assignmain/q2/dense_2/kernel/Adam_1save_3/RestoreV2:67*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@main/q2/dense_2/kernel*
validate_shape(
�
save_3/Assign_68Assigntarget/pi/dense/biassave_3/RestoreV2:68*
use_locking(*
T0*'
_class
loc:@target/pi/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_69Assigntarget/pi/dense/kernelsave_3/RestoreV2:69*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@target/pi/dense/kernel*
validate_shape(
�
save_3/Assign_70Assigntarget/pi/dense_1/biassave_3/RestoreV2:70*
use_locking(*
T0*)
_class
loc:@target/pi/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_71Assigntarget/pi/dense_1/kernelsave_3/RestoreV2:71*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_1/kernel*
validate_shape(*
_output_shapes

:@@
�
save_3/Assign_72Assigntarget/pi/dense_2/biassave_3/RestoreV2:72*
use_locking(*
T0*)
_class
loc:@target/pi/dense_2/bias*
validate_shape(*
_output_shapes
:
�
save_3/Assign_73Assigntarget/pi/dense_2/kernelsave_3/RestoreV2:73*
use_locking(*
T0*+
_class!
loc:@target/pi/dense_2/kernel*
validate_shape(*
_output_shapes

:@
�
save_3/Assign_74Assigntarget/q1/dense/biassave_3/RestoreV2:74*'
_class
loc:@target/q1/dense/bias*
validate_shape(*
_output_shapes
:@*
use_locking(*
T0
�
save_3/Assign_75Assigntarget/q1/dense/kernelsave_3/RestoreV2:75*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*)
_class
loc:@target/q1/dense/kernel
�
save_3/Assign_76Assigntarget/q1/dense_1/biassave_3/RestoreV2:76*
T0*)
_class
loc:@target/q1/dense_1/bias*
validate_shape(*
_output_shapes
:@*
use_locking(
�
save_3/Assign_77Assigntarget/q1/dense_1/kernelsave_3/RestoreV2:77*
_output_shapes

:@@*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_1/kernel*
validate_shape(
�
save_3/Assign_78Assigntarget/q1/dense_2/biassave_3/RestoreV2:78*
T0*)
_class
loc:@target/q1/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_3/Assign_79Assigntarget/q1/dense_2/kernelsave_3/RestoreV2:79*
validate_shape(*
_output_shapes

:@*
use_locking(*
T0*+
_class!
loc:@target/q1/dense_2/kernel
�
save_3/Assign_80Assigntarget/q2/dense/biassave_3/RestoreV2:80*
use_locking(*
T0*'
_class
loc:@target/q2/dense/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_81Assigntarget/q2/dense/kernelsave_3/RestoreV2:81*
T0*)
_class
loc:@target/q2/dense/kernel*
validate_shape(*
_output_shapes

:@*
use_locking(
�
save_3/Assign_82Assigntarget/q2/dense_1/biassave_3/RestoreV2:82*
use_locking(*
T0*)
_class
loc:@target/q2/dense_1/bias*
validate_shape(*
_output_shapes
:@
�
save_3/Assign_83Assigntarget/q2/dense_1/kernelsave_3/RestoreV2:83*+
_class!
loc:@target/q2/dense_1/kernel*
validate_shape(*
_output_shapes

:@@*
use_locking(*
T0
�
save_3/Assign_84Assigntarget/q2/dense_2/biassave_3/RestoreV2:84*
T0*)
_class
loc:@target/q2/dense_2/bias*
validate_shape(*
_output_shapes
:*
use_locking(
�
save_3/Assign_85Assigntarget/q2/dense_2/kernelsave_3/RestoreV2:85*
_output_shapes

:@*
use_locking(*
T0*+
_class!
loc:@target/q2/dense_2/kernel*
validate_shape(
�
save_3/restore_shardNoOp^save_3/Assign^save_3/Assign_1^save_3/Assign_10^save_3/Assign_11^save_3/Assign_12^save_3/Assign_13^save_3/Assign_14^save_3/Assign_15^save_3/Assign_16^save_3/Assign_17^save_3/Assign_18^save_3/Assign_19^save_3/Assign_2^save_3/Assign_20^save_3/Assign_21^save_3/Assign_22^save_3/Assign_23^save_3/Assign_24^save_3/Assign_25^save_3/Assign_26^save_3/Assign_27^save_3/Assign_28^save_3/Assign_29^save_3/Assign_3^save_3/Assign_30^save_3/Assign_31^save_3/Assign_32^save_3/Assign_33^save_3/Assign_34^save_3/Assign_35^save_3/Assign_36^save_3/Assign_37^save_3/Assign_38^save_3/Assign_39^save_3/Assign_4^save_3/Assign_40^save_3/Assign_41^save_3/Assign_42^save_3/Assign_43^save_3/Assign_44^save_3/Assign_45^save_3/Assign_46^save_3/Assign_47^save_3/Assign_48^save_3/Assign_49^save_3/Assign_5^save_3/Assign_50^save_3/Assign_51^save_3/Assign_52^save_3/Assign_53^save_3/Assign_54^save_3/Assign_55^save_3/Assign_56^save_3/Assign_57^save_3/Assign_58^save_3/Assign_59^save_3/Assign_6^save_3/Assign_60^save_3/Assign_61^save_3/Assign_62^save_3/Assign_63^save_3/Assign_64^save_3/Assign_65^save_3/Assign_66^save_3/Assign_67^save_3/Assign_68^save_3/Assign_69^save_3/Assign_7^save_3/Assign_70^save_3/Assign_71^save_3/Assign_72^save_3/Assign_73^save_3/Assign_74^save_3/Assign_75^save_3/Assign_76^save_3/Assign_77^save_3/Assign_78^save_3/Assign_79^save_3/Assign_8^save_3/Assign_80^save_3/Assign_81^save_3/Assign_82^save_3/Assign_83^save_3/Assign_84^save_3/Assign_85^save_3/Assign_9
1
save_3/restore_allNoOp^save_3/restore_shard "�B
save_3/Const:0save_3/Identity:0save_3/restore_all (5 @F8",
train_op 

Adam
Adam_1
Adam_2
Adam_3"�[
	variables�[�[
N
log_alpha:0log_alpha/Assignlog_alpha/read:02log_alpha/initial_value:08
�
main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:021main/pi/dense/kernel/Initializer/random_uniform:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08
�
main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:023main/pi/dense_1/kernel/Initializer/random_uniform:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08
�
main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08
�
main/q1/dense/kernel:0main/q1/dense/kernel/Assignmain/q1/dense/kernel/read:021main/q1/dense/kernel/Initializer/random_uniform:08
v
main/q1/dense/bias:0main/q1/dense/bias/Assignmain/q1/dense/bias/read:02&main/q1/dense/bias/Initializer/zeros:08
�
main/q1/dense_1/kernel:0main/q1/dense_1/kernel/Assignmain/q1/dense_1/kernel/read:023main/q1/dense_1/kernel/Initializer/random_uniform:08
~
main/q1/dense_1/bias:0main/q1/dense_1/bias/Assignmain/q1/dense_1/bias/read:02(main/q1/dense_1/bias/Initializer/zeros:08
�
main/q1/dense_2/kernel:0main/q1/dense_2/kernel/Assignmain/q1/dense_2/kernel/read:023main/q1/dense_2/kernel/Initializer/random_uniform:08
~
main/q1/dense_2/bias:0main/q1/dense_2/bias/Assignmain/q1/dense_2/bias/read:02(main/q1/dense_2/bias/Initializer/zeros:08
�
main/q2/dense/kernel:0main/q2/dense/kernel/Assignmain/q2/dense/kernel/read:021main/q2/dense/kernel/Initializer/random_uniform:08
v
main/q2/dense/bias:0main/q2/dense/bias/Assignmain/q2/dense/bias/read:02&main/q2/dense/bias/Initializer/zeros:08
�
main/q2/dense_1/kernel:0main/q2/dense_1/kernel/Assignmain/q2/dense_1/kernel/read:023main/q2/dense_1/kernel/Initializer/random_uniform:08
~
main/q2/dense_1/bias:0main/q2/dense_1/bias/Assignmain/q2/dense_1/bias/read:02(main/q2/dense_1/bias/Initializer/zeros:08
�
main/q2/dense_2/kernel:0main/q2/dense_2/kernel/Assignmain/q2/dense_2/kernel/read:023main/q2/dense_2/kernel/Initializer/random_uniform:08
~
main/q2/dense_2/bias:0main/q2/dense_2/bias/Assignmain/q2/dense_2/bias/read:02(main/q2/dense_2/bias/Initializer/zeros:08
�
target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:023target/pi/dense/kernel/Initializer/random_uniform:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08
�
target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:025target/pi/dense_1/kernel/Initializer/random_uniform:08
�
target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08
�
target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08
�
target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08
�
target/q1/dense/kernel:0target/q1/dense/kernel/Assigntarget/q1/dense/kernel/read:023target/q1/dense/kernel/Initializer/random_uniform:08
~
target/q1/dense/bias:0target/q1/dense/bias/Assigntarget/q1/dense/bias/read:02(target/q1/dense/bias/Initializer/zeros:08
�
target/q1/dense_1/kernel:0target/q1/dense_1/kernel/Assigntarget/q1/dense_1/kernel/read:025target/q1/dense_1/kernel/Initializer/random_uniform:08
�
target/q1/dense_1/bias:0target/q1/dense_1/bias/Assigntarget/q1/dense_1/bias/read:02*target/q1/dense_1/bias/Initializer/zeros:08
�
target/q1/dense_2/kernel:0target/q1/dense_2/kernel/Assigntarget/q1/dense_2/kernel/read:025target/q1/dense_2/kernel/Initializer/random_uniform:08
�
target/q1/dense_2/bias:0target/q1/dense_2/bias/Assigntarget/q1/dense_2/bias/read:02*target/q1/dense_2/bias/Initializer/zeros:08
�
target/q2/dense/kernel:0target/q2/dense/kernel/Assigntarget/q2/dense/kernel/read:023target/q2/dense/kernel/Initializer/random_uniform:08
~
target/q2/dense/bias:0target/q2/dense/bias/Assigntarget/q2/dense/bias/read:02(target/q2/dense/bias/Initializer/zeros:08
�
target/q2/dense_1/kernel:0target/q2/dense_1/kernel/Assigntarget/q2/dense_1/kernel/read:025target/q2/dense_1/kernel/Initializer/random_uniform:08
�
target/q2/dense_1/bias:0target/q2/dense_1/bias/Assigntarget/q2/dense_1/bias/read:02*target/q2/dense_1/bias/Initializer/zeros:08
�
target/q2/dense_2/kernel:0target/q2/dense_2/kernel/Assigntarget/q2/dense_2/kernel/read:025target/q2/dense_2/kernel/Initializer/random_uniform:08
�
target/q2/dense_2/bias:0target/q2/dense_2/bias/Assigntarget/q2/dense_2/bias/read:02*target/q2/dense_2/bias/Initializer/zeros:08
:
R:0R/AssignR/read:02R/Initializer/random_normal:08
T
beta1_power:0beta1_power/Assignbeta1_power/read:02beta1_power/initial_value:0
T
beta2_power:0beta2_power/Assignbeta2_power/read:02beta2_power/initial_value:0
D
R/Adam:0R/Adam/AssignR/Adam/read:02R/Adam/Initializer/zeros:0
L

R/Adam_1:0R/Adam_1/AssignR/Adam_1/read:02R/Adam_1/Initializer/zeros:0
\
beta1_power_1:0beta1_power_1/Assignbeta1_power_1/read:02beta1_power_1/initial_value:0
\
beta2_power_1:0beta2_power_1/Assignbeta2_power_1/read:02beta2_power_1/initial_value:0
�
main/pi/dense/kernel/Adam:0 main/pi/dense/kernel/Adam/Assign main/pi/dense/kernel/Adam/read:02-main/pi/dense/kernel/Adam/Initializer/zeros:0
�
main/pi/dense/kernel/Adam_1:0"main/pi/dense/kernel/Adam_1/Assign"main/pi/dense/kernel/Adam_1/read:02/main/pi/dense/kernel/Adam_1/Initializer/zeros:0
�
main/pi/dense/bias/Adam:0main/pi/dense/bias/Adam/Assignmain/pi/dense/bias/Adam/read:02+main/pi/dense/bias/Adam/Initializer/zeros:0
�
main/pi/dense/bias/Adam_1:0 main/pi/dense/bias/Adam_1/Assign main/pi/dense/bias/Adam_1/read:02-main/pi/dense/bias/Adam_1/Initializer/zeros:0
�
main/pi/dense_1/kernel/Adam:0"main/pi/dense_1/kernel/Adam/Assign"main/pi/dense_1/kernel/Adam/read:02/main/pi/dense_1/kernel/Adam/Initializer/zeros:0
�
main/pi/dense_1/kernel/Adam_1:0$main/pi/dense_1/kernel/Adam_1/Assign$main/pi/dense_1/kernel/Adam_1/read:021main/pi/dense_1/kernel/Adam_1/Initializer/zeros:0
�
main/pi/dense_1/bias/Adam:0 main/pi/dense_1/bias/Adam/Assign main/pi/dense_1/bias/Adam/read:02-main/pi/dense_1/bias/Adam/Initializer/zeros:0
�
main/pi/dense_1/bias/Adam_1:0"main/pi/dense_1/bias/Adam_1/Assign"main/pi/dense_1/bias/Adam_1/read:02/main/pi/dense_1/bias/Adam_1/Initializer/zeros:0
�
main/pi/dense_2/kernel/Adam:0"main/pi/dense_2/kernel/Adam/Assign"main/pi/dense_2/kernel/Adam/read:02/main/pi/dense_2/kernel/Adam/Initializer/zeros:0
�
main/pi/dense_2/kernel/Adam_1:0$main/pi/dense_2/kernel/Adam_1/Assign$main/pi/dense_2/kernel/Adam_1/read:021main/pi/dense_2/kernel/Adam_1/Initializer/zeros:0
�
main/pi/dense_2/bias/Adam:0 main/pi/dense_2/bias/Adam/Assign main/pi/dense_2/bias/Adam/read:02-main/pi/dense_2/bias/Adam/Initializer/zeros:0
�
main/pi/dense_2/bias/Adam_1:0"main/pi/dense_2/bias/Adam_1/Assign"main/pi/dense_2/bias/Adam_1/read:02/main/pi/dense_2/bias/Adam_1/Initializer/zeros:0
\
beta1_power_2:0beta1_power_2/Assignbeta1_power_2/read:02beta1_power_2/initial_value:0
\
beta2_power_2:0beta2_power_2/Assignbeta2_power_2/read:02beta2_power_2/initial_value:0
�
main/q1/dense/kernel/Adam:0 main/q1/dense/kernel/Adam/Assign main/q1/dense/kernel/Adam/read:02-main/q1/dense/kernel/Adam/Initializer/zeros:0
�
main/q1/dense/kernel/Adam_1:0"main/q1/dense/kernel/Adam_1/Assign"main/q1/dense/kernel/Adam_1/read:02/main/q1/dense/kernel/Adam_1/Initializer/zeros:0
�
main/q1/dense/bias/Adam:0main/q1/dense/bias/Adam/Assignmain/q1/dense/bias/Adam/read:02+main/q1/dense/bias/Adam/Initializer/zeros:0
�
main/q1/dense/bias/Adam_1:0 main/q1/dense/bias/Adam_1/Assign main/q1/dense/bias/Adam_1/read:02-main/q1/dense/bias/Adam_1/Initializer/zeros:0
�
main/q1/dense_1/kernel/Adam:0"main/q1/dense_1/kernel/Adam/Assign"main/q1/dense_1/kernel/Adam/read:02/main/q1/dense_1/kernel/Adam/Initializer/zeros:0
�
main/q1/dense_1/kernel/Adam_1:0$main/q1/dense_1/kernel/Adam_1/Assign$main/q1/dense_1/kernel/Adam_1/read:021main/q1/dense_1/kernel/Adam_1/Initializer/zeros:0
�
main/q1/dense_1/bias/Adam:0 main/q1/dense_1/bias/Adam/Assign main/q1/dense_1/bias/Adam/read:02-main/q1/dense_1/bias/Adam/Initializer/zeros:0
�
main/q1/dense_1/bias/Adam_1:0"main/q1/dense_1/bias/Adam_1/Assign"main/q1/dense_1/bias/Adam_1/read:02/main/q1/dense_1/bias/Adam_1/Initializer/zeros:0
�
main/q1/dense_2/kernel/Adam:0"main/q1/dense_2/kernel/Adam/Assign"main/q1/dense_2/kernel/Adam/read:02/main/q1/dense_2/kernel/Adam/Initializer/zeros:0
�
main/q1/dense_2/kernel/Adam_1:0$main/q1/dense_2/kernel/Adam_1/Assign$main/q1/dense_2/kernel/Adam_1/read:021main/q1/dense_2/kernel/Adam_1/Initializer/zeros:0
�
main/q1/dense_2/bias/Adam:0 main/q1/dense_2/bias/Adam/Assign main/q1/dense_2/bias/Adam/read:02-main/q1/dense_2/bias/Adam/Initializer/zeros:0
�
main/q1/dense_2/bias/Adam_1:0"main/q1/dense_2/bias/Adam_1/Assign"main/q1/dense_2/bias/Adam_1/read:02/main/q1/dense_2/bias/Adam_1/Initializer/zeros:0
�
main/q2/dense/kernel/Adam:0 main/q2/dense/kernel/Adam/Assign main/q2/dense/kernel/Adam/read:02-main/q2/dense/kernel/Adam/Initializer/zeros:0
�
main/q2/dense/kernel/Adam_1:0"main/q2/dense/kernel/Adam_1/Assign"main/q2/dense/kernel/Adam_1/read:02/main/q2/dense/kernel/Adam_1/Initializer/zeros:0
�
main/q2/dense/bias/Adam:0main/q2/dense/bias/Adam/Assignmain/q2/dense/bias/Adam/read:02+main/q2/dense/bias/Adam/Initializer/zeros:0
�
main/q2/dense/bias/Adam_1:0 main/q2/dense/bias/Adam_1/Assign main/q2/dense/bias/Adam_1/read:02-main/q2/dense/bias/Adam_1/Initializer/zeros:0
�
main/q2/dense_1/kernel/Adam:0"main/q2/dense_1/kernel/Adam/Assign"main/q2/dense_1/kernel/Adam/read:02/main/q2/dense_1/kernel/Adam/Initializer/zeros:0
�
main/q2/dense_1/kernel/Adam_1:0$main/q2/dense_1/kernel/Adam_1/Assign$main/q2/dense_1/kernel/Adam_1/read:021main/q2/dense_1/kernel/Adam_1/Initializer/zeros:0
�
main/q2/dense_1/bias/Adam:0 main/q2/dense_1/bias/Adam/Assign main/q2/dense_1/bias/Adam/read:02-main/q2/dense_1/bias/Adam/Initializer/zeros:0
�
main/q2/dense_1/bias/Adam_1:0"main/q2/dense_1/bias/Adam_1/Assign"main/q2/dense_1/bias/Adam_1/read:02/main/q2/dense_1/bias/Adam_1/Initializer/zeros:0
�
main/q2/dense_2/kernel/Adam:0"main/q2/dense_2/kernel/Adam/Assign"main/q2/dense_2/kernel/Adam/read:02/main/q2/dense_2/kernel/Adam/Initializer/zeros:0
�
main/q2/dense_2/kernel/Adam_1:0$main/q2/dense_2/kernel/Adam_1/Assign$main/q2/dense_2/kernel/Adam_1/read:021main/q2/dense_2/kernel/Adam_1/Initializer/zeros:0
�
main/q2/dense_2/bias/Adam:0 main/q2/dense_2/bias/Adam/Assign main/q2/dense_2/bias/Adam/read:02-main/q2/dense_2/bias/Adam/Initializer/zeros:0
�
main/q2/dense_2/bias/Adam_1:0"main/q2/dense_2/bias/Adam_1/Assign"main/q2/dense_2/bias/Adam_1/read:02/main/q2/dense_2/bias/Adam_1/Initializer/zeros:0
\
beta1_power_3:0beta1_power_3/Assignbeta1_power_3/read:02beta1_power_3/initial_value:0
\
beta2_power_3:0beta2_power_3/Assignbeta2_power_3/read:02beta2_power_3/initial_value:0
d
log_alpha/Adam:0log_alpha/Adam/Assignlog_alpha/Adam/read:02"log_alpha/Adam/Initializer/zeros:0
l
log_alpha/Adam_1:0log_alpha/Adam_1/Assignlog_alpha/Adam_1/read:02$log_alpha/Adam_1/Initializer/zeros:0"�(
trainable_variables�(�(
N
log_alpha:0log_alpha/Assignlog_alpha/read:02log_alpha/initial_value:08
�
main/pi/dense/kernel:0main/pi/dense/kernel/Assignmain/pi/dense/kernel/read:021main/pi/dense/kernel/Initializer/random_uniform:08
v
main/pi/dense/bias:0main/pi/dense/bias/Assignmain/pi/dense/bias/read:02&main/pi/dense/bias/Initializer/zeros:08
�
main/pi/dense_1/kernel:0main/pi/dense_1/kernel/Assignmain/pi/dense_1/kernel/read:023main/pi/dense_1/kernel/Initializer/random_uniform:08
~
main/pi/dense_1/bias:0main/pi/dense_1/bias/Assignmain/pi/dense_1/bias/read:02(main/pi/dense_1/bias/Initializer/zeros:08
�
main/pi/dense_2/kernel:0main/pi/dense_2/kernel/Assignmain/pi/dense_2/kernel/read:023main/pi/dense_2/kernel/Initializer/random_uniform:08
~
main/pi/dense_2/bias:0main/pi/dense_2/bias/Assignmain/pi/dense_2/bias/read:02(main/pi/dense_2/bias/Initializer/zeros:08
�
main/q1/dense/kernel:0main/q1/dense/kernel/Assignmain/q1/dense/kernel/read:021main/q1/dense/kernel/Initializer/random_uniform:08
v
main/q1/dense/bias:0main/q1/dense/bias/Assignmain/q1/dense/bias/read:02&main/q1/dense/bias/Initializer/zeros:08
�
main/q1/dense_1/kernel:0main/q1/dense_1/kernel/Assignmain/q1/dense_1/kernel/read:023main/q1/dense_1/kernel/Initializer/random_uniform:08
~
main/q1/dense_1/bias:0main/q1/dense_1/bias/Assignmain/q1/dense_1/bias/read:02(main/q1/dense_1/bias/Initializer/zeros:08
�
main/q1/dense_2/kernel:0main/q1/dense_2/kernel/Assignmain/q1/dense_2/kernel/read:023main/q1/dense_2/kernel/Initializer/random_uniform:08
~
main/q1/dense_2/bias:0main/q1/dense_2/bias/Assignmain/q1/dense_2/bias/read:02(main/q1/dense_2/bias/Initializer/zeros:08
�
main/q2/dense/kernel:0main/q2/dense/kernel/Assignmain/q2/dense/kernel/read:021main/q2/dense/kernel/Initializer/random_uniform:08
v
main/q2/dense/bias:0main/q2/dense/bias/Assignmain/q2/dense/bias/read:02&main/q2/dense/bias/Initializer/zeros:08
�
main/q2/dense_1/kernel:0main/q2/dense_1/kernel/Assignmain/q2/dense_1/kernel/read:023main/q2/dense_1/kernel/Initializer/random_uniform:08
~
main/q2/dense_1/bias:0main/q2/dense_1/bias/Assignmain/q2/dense_1/bias/read:02(main/q2/dense_1/bias/Initializer/zeros:08
�
main/q2/dense_2/kernel:0main/q2/dense_2/kernel/Assignmain/q2/dense_2/kernel/read:023main/q2/dense_2/kernel/Initializer/random_uniform:08
~
main/q2/dense_2/bias:0main/q2/dense_2/bias/Assignmain/q2/dense_2/bias/read:02(main/q2/dense_2/bias/Initializer/zeros:08
�
target/pi/dense/kernel:0target/pi/dense/kernel/Assigntarget/pi/dense/kernel/read:023target/pi/dense/kernel/Initializer/random_uniform:08
~
target/pi/dense/bias:0target/pi/dense/bias/Assigntarget/pi/dense/bias/read:02(target/pi/dense/bias/Initializer/zeros:08
�
target/pi/dense_1/kernel:0target/pi/dense_1/kernel/Assigntarget/pi/dense_1/kernel/read:025target/pi/dense_1/kernel/Initializer/random_uniform:08
�
target/pi/dense_1/bias:0target/pi/dense_1/bias/Assigntarget/pi/dense_1/bias/read:02*target/pi/dense_1/bias/Initializer/zeros:08
�
target/pi/dense_2/kernel:0target/pi/dense_2/kernel/Assigntarget/pi/dense_2/kernel/read:025target/pi/dense_2/kernel/Initializer/random_uniform:08
�
target/pi/dense_2/bias:0target/pi/dense_2/bias/Assigntarget/pi/dense_2/bias/read:02*target/pi/dense_2/bias/Initializer/zeros:08
�
target/q1/dense/kernel:0target/q1/dense/kernel/Assigntarget/q1/dense/kernel/read:023target/q1/dense/kernel/Initializer/random_uniform:08
~
target/q1/dense/bias:0target/q1/dense/bias/Assigntarget/q1/dense/bias/read:02(target/q1/dense/bias/Initializer/zeros:08
�
target/q1/dense_1/kernel:0target/q1/dense_1/kernel/Assigntarget/q1/dense_1/kernel/read:025target/q1/dense_1/kernel/Initializer/random_uniform:08
�
target/q1/dense_1/bias:0target/q1/dense_1/bias/Assigntarget/q1/dense_1/bias/read:02*target/q1/dense_1/bias/Initializer/zeros:08
�
target/q1/dense_2/kernel:0target/q1/dense_2/kernel/Assigntarget/q1/dense_2/kernel/read:025target/q1/dense_2/kernel/Initializer/random_uniform:08
�
target/q1/dense_2/bias:0target/q1/dense_2/bias/Assigntarget/q1/dense_2/bias/read:02*target/q1/dense_2/bias/Initializer/zeros:08
�
target/q2/dense/kernel:0target/q2/dense/kernel/Assigntarget/q2/dense/kernel/read:023target/q2/dense/kernel/Initializer/random_uniform:08
~
target/q2/dense/bias:0target/q2/dense/bias/Assigntarget/q2/dense/bias/read:02(target/q2/dense/bias/Initializer/zeros:08
�
target/q2/dense_1/kernel:0target/q2/dense_1/kernel/Assigntarget/q2/dense_1/kernel/read:025target/q2/dense_1/kernel/Initializer/random_uniform:08
�
target/q2/dense_1/bias:0target/q2/dense_1/bias/Assigntarget/q2/dense_1/bias/read:02*target/q2/dense_1/bias/Initializer/zeros:08
�
target/q2/dense_2/kernel:0target/q2/dense_2/kernel/Assigntarget/q2/dense_2/kernel/read:025target/q2/dense_2/kernel/Initializer/random_uniform:08
�
target/q2/dense_2/bias:0target/q2/dense_2/bias/Assigntarget/q2/dense_2/bias/read:02*target/q2/dense_2/bias/Initializer/zeros:08
:
R:0R/AssignR/read:02R/Initializer/random_normal:08*�
serving_default�
)
x$
Placeholder:0���������
+
a&
Placeholder_1:0��������� 
q2
Sum_4:0���������?
pi9
&main/pi/Categorical/sample/Reshape_1:0��������� 
q1
Sum_3:0���������tensorflow/serving/predict