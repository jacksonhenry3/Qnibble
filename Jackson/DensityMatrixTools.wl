(* ::Package:: *)

(* ::Text:: *)
(*TODO*)
(*define equality that takes order in to account*)
(*define multiplication taht can do both tensor and dot porducts.*)


(* ::Section:: *)
(*Error Codes*)


qBitMissmatch::oob="`1` and `2` must be the same.";
qBitduplicate::oob="`1` must have unique names and unique values";
qBitvalue::mismatch="`1` must only refrence values in 1...n";
dimensionMismatch::value="\!\(\*SuperscriptBox[\(2\), \(nqbits\)]\) must be equal to the size of the matrix instead of nqbits = `1` and matrixdim = `2`";


(* ::Subsection:: *)
(*Define the DensityMatrix (DM) object*)


(* ::Text:: *)
(*TODO*)
(*Check that qids is unique in both key and value!*)


SetAttributes[MakeDM, HoldRest]
MakeDM[matrix_, qIDs_] :=
    Module[{i,j,IDmap},
    If[!DuplicateFreeQ[Keys[qIDs]],
    Message[qBitduplicate::oob,qIDs];Return[]];
    
    If[Sort[Values[qIDs]]=!=Table[i,{i,Length[qIDs]}],
    Message[qBitvalue::mismatch,qIDs];Return[]];
    
    If[Length[matrix]=!=2^Length[qIDs],
    Message[dimensionMismatch::value,Length[qIDs],Length[matrix]];Return[]];
    
    
    
    DM[<|data -> SparseArray[matrix], qbitIDs -> qIDs , nqbit ->  Length[qIDs]|>]
    ]
MakeDM[matrix_]:=MakeDM[matrix,Association[Table[Subscript[\[ScriptCapitalQ], i]-> i,{i,Log2[Length[matrix]]}]]]


(* ::Subsection:: *)
(*Utility functions*)


values[nqbit_]:=
	Transpose[Table[IntegerDigits[i, 2, nqbit], {i, 0, 2^nqbit - 1}]]


Simplify[DM[\[Rho]_]]^:=MakeDM[\[Rho][data]//Simplify, \[Rho][qbitIDs]]


Inc[a_,n_:1]:=Association[Table[i-> a[i]+n,{i,Keys[a]}]]


(* ::Subsection:: *)
(*Initialize thermal density matrices*)


ThermalQBit[p_, i_:1] :=
    MakeDM[{{1 - p, 0}, {0, p}}, <|i-> 1|>]
NThermalQBit[probs_,indices_] :=
    Module[{c},
        Prod @@ Table[ThermalQBit[probs[[i]], indices[[i]]], {i, Length[probs]}]
    ]
    NThermalQBit[probs_] :=   NThermalQBit[probs,Table[Subscript[\[ScriptCapitalQ], j], {j, Length[probs]}]
    ]


(* ::Subsection:: *)
(*Random Energy preserving Hamiltonian*)


MakeHamiltonian[i1_,i2_,n_]:=MakeDM[SparseArray[{{i1,i2}-> I,{i2,i1}->-I},2^n]]


RandomHamiltonian[Energy_,nqbits_]:=Module[{protoState,perms,indeces},
protoState = Table[If[i<Energy+1,1,0],{i,nqbits}];
perms = Permutations[protoState];
indeces = FromDigits[#,2]&/@perms;
Plus@@Flatten[Table[Table[A[i+1,j+1]MakeHamiltonian[indeces[[i]],indeces[[j]],nqbits],{j,i-1}],{i,Length[perms]}],1]
]


(* ::Subsection:: *)
(*Basic Operations*)


DM[\[Rho]_][data] ^:= \[Rho][data];
DM[\[Rho]_][qbitIDs] ^:= \[Rho][qbitIDs];


DM[a_] \[TensorProduct] DM[b_] ^:= MakeDM[KroneckerProduct[a[data], b[data]], Merge[{a[qbitIDs], Inc[b[qbitIDs], Length[a[qbitIDs]]]}, #[[1]]&]]
Prod[a_, b__] :=
	a \[TensorProduct] b

DM[a_] . DM[b_] ^:= Module[{},
	If[a[qbitIDs] != b[qbitIDs],
		Message[qBitMissmatch::oob, a[qbitIDs], b[qbitIDs]];
		Return[];
	];
	MakeDM[a[data] . b[data], a[qbitIDs]]
]

b_ DM[a_] ^:= DM[<|data-> a[data]b,qbitIDs-> a[qbitIDs], nqbit-> a[nqbit]|>]

DM[a_] + DM[b_] ^:= Module[{},
	If[a[qbitIDs] != b[qbitIDs],
		Message[qBitMissmatch::oob, a[qbitIDs], b[qbitIDs]];
		Return[];
	];
	DM[<|data-> a[data] + b[data],qbitIDs-> a[qbitIDs], nqbit-> a[nqbit]|>]
]


Tr[DM[a_]] ^:= Tr[a[data]]
Transpose[DM[a_]]^:=MakeDM[Transpose[a[data]], a[qbitIDs]]


ReOrder[DM[\[Rho]_], newOrder_] :=
	Module[{n = \[Rho][nqbit], ids = Keys[\[Rho][qbitIDs]], map, V, L1, L2, rules, newRules},
		If[Sort[Keys[newOrder]] =!= Sort[ids] || Sort[Values[newOrder]] =!= Sort[Values[\[Rho][qbitIDs]]],
			Message[qBitMissmatch::oob, \[Rho][qbitIDs], newOrder];
			Return[];
		];
		V = values[n];
		L1 = FromDigits[#, 2] + 1& /@ Transpose[Table[V[[\[Rho][qbitIDs][id]]], {id, ids}]];
		L2 = FromDigits[#, 2] + 1& /@ Transpose[Table[V[[newOrder[id]]], {id, ids}]];
		map = Association[Table[L1[[i]] -> L2[[i]], {i, Length[L1]}]];
		rules = ArrayRules[\[Rho][data]];
		(*The "Dua Lipa" step*)
		newRules = Table[(rules[[i, 1]] /. map) -> (rules[[i, 2]]), {i, Length[rules]}];
		MakeDM[SparseArray[newRules], newOrder]
	]


PartialTrace[DM[\[Rho]_], lst_] :=
	Module[{M = \[Rho][data], d = Dimensions[\[Rho][data]], l, m, dist, newlst, h1, h2, n, res, newQBitIds, newQBitIdsPrime},
		If[lst == {},
			Return[DM[\[Rho]]]
		];
		n = \[Rho][qbitIDs][Last[lst]];
		newlst = Drop[lst, -1];
		l = d[[1]];
		dist = values[\[Rho][nqbit]];
		{h1, h2} = {Position[dist[[n]], 0] // Flatten, Position[dist[[n]], 1] // Flatten};
		newQBitIdsPrime = KeyDrop[\[Rho][qbitIDs], Last[lst]];
		newQBitIds = Association[
			Table[If[newQBitIdsPrime[id] > n,
				id -> newQBitIdsPrime[id] - 1
				,
				id -> newQBitIdsPrime[id]
			],
			{id, Keys[newQBitIdsPrime]}]
		];
		res = MakeDM[(M[[h1, h1]] + M[[h2, h2]]), newQBitIds];
		PartialTrace[res, newlst]
	]
PTR = PartialTrace;


(* ::Subsection:: *)
(*Vizualization*)


Show[DM[a_]]^:=MatrixPlot[a[data],Frame->False]


Inspect[DM[a_]]:=a[data]//Simplify//MatrixForm
