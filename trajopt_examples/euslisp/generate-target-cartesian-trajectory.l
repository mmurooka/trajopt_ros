
(defun dump-circle-trajectory
    ()
  (with-open-file
   (f "/tmp/cart_traj.csv" :direction :output :if-exists :new-version :if-does-not-exist :create)
   (format f " Line 1,,,,,,~%")
   (format f " Line 2,,,,,,~%")
   (let* ((radius 500) ;; [mm]
          (divide-num 4)
          (min-theta (deg2rad -60))
          (max-theta (deg2rad -30))
          (pos-offset (float-vector 0 0 100))
          )
     (dotimes (i divide-num)
       (let* ((theta (+ (* (- max-theta min-theta) (/ (float i) divide-num)) min-theta))
              (pos (v+ (scale radius (float-vector (cos theta) (sin theta) 0)) pos-offset))
              (normal (normalize-vector (float-vector 0 0 -1)))
              )
         (format f "~a,~a,~a,~a,~a,~a,~a~%"
                 (1+ i) (elt pos 0) (elt pos 1) (elt pos 2) (elt normal 0) (elt normal 1) (elt normal 2))
         ))))
  )
(warn "(dump-circle-trajectory)~%")
