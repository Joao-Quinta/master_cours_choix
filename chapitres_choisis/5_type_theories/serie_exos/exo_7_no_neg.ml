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
  if n = 0 then true
  else odd (n - 1)
and odd n = 
  if n = 0 then false
  else even (n - 1)
in
even(4)