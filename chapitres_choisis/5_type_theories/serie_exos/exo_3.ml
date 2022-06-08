(* Ceci est un éditeur pour OCaml
   Entrez votre programme ici, et envoyez-le au toplevel en utilisant le
   bouton "Évaluer le code" ci-dessous ou [Ctrl-e]. *)

let l = 1 :: 2 :: 3 :: [] in
match l with 
| f :: s :: t :: r -> t 
| _ -> 0 ;;

let l1 = ('c', true, 4.5) in
match l1 with 
| f , s , t -> t ;;