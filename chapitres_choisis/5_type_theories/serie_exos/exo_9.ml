(* Ceci est un éditeur pour OCaml
   Entrez votre programme ici, et envoyez-le au toplevel en utilisant le
   bouton "Évaluer le code" ci-dessous ou [Ctrl-e]. *)

let rec filter f l = 
  match l with
  | hd :: tl -> 
      if f hd then hd :: filter f tl
      else filter f tl
  | [] -> [] 
in
filter (fun x -> x < 5)(1 :: 10 :: 4 :: [])