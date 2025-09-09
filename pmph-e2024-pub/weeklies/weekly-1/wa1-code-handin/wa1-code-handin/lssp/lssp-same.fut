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
-- compiled input {
--    [3, 2, 1, 2, 3, 4, 1, 2, 3, 4, 2, 2, 2, 1, 2, 3, 4, 5]
-- }  
-- output { 
--    3
-- }
-- compiled input {
--    [1, 4, 1, 2, 0, 0, 2, 4, 5, 0, 0, 0, 0, 1, 2, 0]
-- }
-- output { 
--    4
-- }
--
-- compiled random input { [100000000]i32 }

import "lssp"
import "lssp-seq"

let main (xs: []i32) : i32 =
  let pred1 _x = true
  let pred2 x y = (x == y)
    -- in  lssp_seq pred1 pred2 xs
  in  lssp pred1 pred2 xs
