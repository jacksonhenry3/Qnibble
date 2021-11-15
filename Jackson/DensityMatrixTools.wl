(* ::Package:: *)

(* ::Text:: *)
(*TODO*)
(*define equality that takes order in to account.*)


(* ::Subsection:: *)
(*Error Codes*)


qBitMissmatch::oob="`1` and `2` must be the same.";
qBitduplicate::oob="`1` must have unique names and unique values";
qBitvalue::mismatch="`1` must refrence values in 1...n exactly once each.";
NotEnoughWater::mismatch="Please drink some water, it's good for you. (also this is too big, must be one qubit)";
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
    
    If[Dimensions[matrix][[1]]=!=2^Length[qIDs],
    Message[dimensionMismatch::value,Length[qIDs],Dimensions[matrix]];Return[]];
    
    
    
    DM[<|data -> SparseArray[matrix], qbitIDs -> qIDs , nqbit ->  Length[qIDs]|>]
    ]
MakeDM[matrix_]:=MakeDM[matrix,Association[Table[Subscript[\[ScriptCapitalQ], i]-> i,{i,Log2[Length[matrix]]}]]]


(* ::Subsection::Closed:: *)
(*Utility functions*)


values[nqbit_]:=
	Transpose[Table[IntegerDigits[i, 2, nqbit], {i, 0, 2^nqbit - 1}]]


Simplify[DM[\[Rho]_]]^:=MakeDM[\[Rho][data]//Simplify, \[Rho][qbitIDs]]


Inc[a_,n_:1]:=Association[Table[i-> a[i]+n,{i,Keys[a]}]]


(* ::Subsection::Closed:: *)
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


MakeHamiltonian[i1_,i2_,n_]:=SparseArray[{{i1,i2}-> I,{i2,i1}->-I},2^n]


RandomHamiltonian[Energy_,nqbits_]:=Module[{protoState,perms,indeces},
protoState = Table[If[i<Energy+1,1,0],{i,nqbits}];
perms = Permutations[protoState];
indeces = FromDigits[#,2]&/@perms;
Plus@@Flatten[Table[Table[Random[]MakeHamiltonian[indeces[[i]]+1,indeces[[j]]+1,nqbits],{j,i-1}],{i,Length[perms]}],1]
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


(* ::Text:: *)
(*TODO: add ReOrder with no order to go back to "Canonical Order"*)
(*TODO: add ReOrder with list arg mapping each to the ith canonical arg.*)


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
		MakeDM[SparseArray[newRules,Dimensions[\[Rho][data]]], newOrder]
	]


PartialTrace[DM[\[Rho]_], lst_] :=
	Module[{M = \[Rho][data], d = Dimensions[\[Rho][data]], l, m, dist, newlst, h1, h2, n, res, newQBitIds, newQBitIdsPrime},
		If[lst == {},
			Return[DM[\[Rho]]]
		];
		(*lst = Flatten[{lst}];*)
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
	];
	
PTR = PartialTrace;


(* ::Subsection::Closed:: *)
(*Visualization*)


Show[DM[a_]]^:=MatrixPlot[a[data],Frame->False]


ArrayShow[DM[a_]]^:=ArrayPlot[Re[a[data]],Frame->False, PlotRange -> {-10^-5,10^-5}]  


Inspect[DM[a_]]:=Module[{n = a[nqbit],labels },
labels = Table[StringJoin[ToString /@IntegerDigits[i-1,2,n]],{i,2^n}];

TableForm[a[data]//Simplify,TableHeadings->{labels,labels},TableAlignments->Center]
]


(* ::Subsection:: *)
(*Thermal properties*)


(* ::Text:: *)
(*TODO: Check for id in qbit ids*)


Temp[DM[\[Rho]_],qbitID_]^:=Module[{p},
p = PartialTrace[DM[\[Rho]],DeleteCases[Keys[\[Rho][qbitIDs]],qbitID]][data][[2,2]];
1/Log[(1-p)/p]//Simplify
]
Temp[DM[\[Rho]_]]^:=Module[{p},If[\[Rho][nqbit]==1,p = \[Rho][data][[2,2]];1/Log[(1-p)/p],Message[NotEnoughWater::mismatch];Return[];]]


AvgTemp[DM[\[Rho]_],qbitIDs_]:=Mean[Table[Temp[DM[\[Rho]],qbit],{qbit,qbitIDs}]]


(* ::Text:: *)
(*TODO: Test for same dimensionality*)
(*TODO: Check that *)


Distance[DM[\[Rho]1_],DM[\[Rho]2_]]^:= Module[{d1,d2},
d1 = \[Rho]1[data];
d2 = \[Rho]2[data];
Tr[d1(MatrixLog[d1]-MatrixLog[d2])]//Quiet
]


(* ::Text:: *)
(*TODO make cute simple version of distance*)


(*D[DM[\[Rho]1_]||DM[\[Rho]2_]]^:=Distance[DM[\[Rho]1],DM[\[Rho]2]]*)


(* ::Text:: *)
(*Add trace distance*)


ExtractableWork[DM[\[Rho]1i_],DM[\[Rho]2i_],DM[\[Rho]1f_],DM[\[Rho]2f_]] ^:= Temp[DM[\[Rho]2f]] Distance[DM[\[Rho]1f],DM[\[Rho]2f]]-Temp[DM[\[Rho]2i]] Distance[DM[\[Rho]1i],DM[\[Rho]2i]]
