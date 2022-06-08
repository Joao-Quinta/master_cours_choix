(* Ceci est un éditeur pour OCaml
   Entrez votre programme ici, et envoyez-le au toplevel en utilisant le
   bouton "Évaluer le code" ci-dessous ou [Ctrl-e]. *)

let rec fibo n = 
  if n < 3 then 
    if n = 0 then 0
    else 1
  else fibo(n - 2) + fibo (n - 1) 
in 
fibo 7
