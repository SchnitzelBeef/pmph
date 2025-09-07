-- Parallel Longest Satisfying Segment


-- ==
-- compiled input {
--    [1i32, -2i32, -2i32, 0i32, 0i32, 0i32, 0i32, 0i32, 3i32, 4i32, -6i32, 1i32]
-- }
-- output {
--    5i32
-- }
-- compiled input {
--    [1, 1, 1, 0, 0, 1, 2, 4, 5, 2, 2, 5, 5]
-- }
-- output {
--    3
-- }
-- "Pattern-200000010-elements" script input { mk_input 10000000i64 }
-- output { 2i32 }

entry mk_input (n:i64) : [20*n+10]i32 =
   let pattern = [-100i32, 10, 3, -1, 4, -1, 5, 1, 1, -100]
   let rep_pattern = replicate n pattern |> flatten
   let max_segment = iota 10 |> map i32.i64
   in  (rep_pattern ++ max_segment ++ rep_pattern) :> [20*n+10]i32

import "lssp"
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
    -- in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
