(* Ceci est un éditeur pour OCaml
   Entrez votre programme ici, et envoyez-le au toplevel en utilisant le
   bouton "Évaluer le code" ci-dessous ou [Ctrl-e]. 

    -> do with 1 function 
      
let rec even n = 
  if n = 0 then true
  else not (even(n - 1)) 
in
even 11*)
let rec even n = 
  match n with
  | 0 -> true
  | m when m < 0 -> odd (m + 1)
  | m -> odd(m-1)
and odd n =
  match n with 
  | 0 -> false
  | m when m < 0 -> even (m + 1)
  | m -> even(m-1)
in
odd 11