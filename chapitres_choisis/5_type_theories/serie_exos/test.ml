let l = 1 :: 2 :: 3 :: [] in
match l with 
| _ :: x :: tl -> x 
| x :: tl -> x
| _ -> 0 
