-- Parallel Longest Satisfying Segment
--
-- ==
-- compiled input {
--    [1i32, -2, -2, 0, 0, 0, 0, 0, 3, 4, -6, 1]
-- }
-- output {
--    5
-- }
-- Added tests:
-- compiled input {
--    [1, 1, 1, 0, 0, 1, 2, 4, 5, 2, 2, 5, 5]
-- }
-- output {
--    2
-- }
-- compiled input {
--    [3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 2, 2, 2, 1, 2, 3, 4, 5]
-- }  
-- output { 
--    0
-- }
-- compiled input {
--    [1, 4, 1, 2, 0, 0, 2, 4, 5, 0, 0, 0, 0, 1, 2, 0]
-- }
-- output { 
--    4
-- }
-- compiled random input { [100000000]i32 }

import "lssp-seq"
import "lssp"

type int = i32

let main (xs: []int) : int =
  let pred1 x   = (x == 0)
  let pred2 x y = (x == 0) && (y == 0)
--  in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
