;;; load dependencies
(ql:quickload :mgl-mat)

;;; define package
(defpackage cl-zerodl
  (:use :cl :mgl-mat)
  (:nicknames :zerodl))

;;; enter the package
(in-package :cl-zerodl)

;;; settings
(setf *default-mat-ctype* :float)
(setf *cuda-enabled* t)

(defparameter ma (make-mat '(2 2) :initial-contents '((1 -2) (-3 4))))
(defparameter mb (make-mat '(2 2) :initial-contents '((5 6) (7 8))))
(defparameter mc (make-mat '(2 2) :initial-element 0.0))

;;; mc = alpha * ma * mb + beta * mc
(gemm! 1.0 ma mb 0.0 mc)

;; update repl-size
(setf *print-length* 10
      *print-length* 10)

(defparameter ma (make-mat '(10000 10000)))
(defparameter mb (make-mat '(10000 10000)))
(defparameter mc (make-mat '(10000 10000)))

;; initialize with gauss distribution
(uniform-random! ma)
(uniform-random! mb)


;; openblas ... ?
(time (gemm! 1.0 ma mb 0.0 mc))

;; cublas
(with-cuda* ()
  (time (gemm! 1.0 ma mb 0.0 mc)))

;;; add or sub operation
(defparameter va (make-mat '(3 1) :initial-contents '((1) (2) (3))))
(defparameter vb (make-mat '(3 1) :initial-contents '((10) (20) (30))))

;; vb = alpha + va + vb
(axpy! 1.0 va vb)


;; Sigmoid
(defun sigmoid! (v)
  (.logistic! v))

(defparameter x (make-mat '(2 1) :initial-contents '((1.0) (0.5))))
(defparameter W1 (make-mat '(3 2) :initial-contents '((0.1 0.2) (0.3 0.4) (0.5 0.6))))
(defparameter b1 (make-mat '(3 1) :initial-contents '((0.1) (0.2) (0.3))))
(defparameter z1 (make-mat '(3 1) :initial-element 0.0))

;; z1 = W1 * x : 3 x 2 2 * 1 = 3 * 1
(gemm! 1.0 W1 x 0.0 z1)
;; z1 = b1 + z1
(axpy! 1.0 b1 z1) ;; => #<MAT 3x1 AF #2A((0.3) (0.7) (1.1))>
(sigmoid! z1) ;; => #<MAT 3x1 ABF #2A((0.5744425) (0.66818774) (0.7502601))>

(defparameter W2 (make-mat '(2 3) :initial-contents '((0.1 0.2 0.3) (0.4 0.5 0.6))))
(defparameter b2 (make-mat '(2 1) :initial-contents '((0.1) (0.2))))
(defparameter z2 (make-mat '(2 1) :initial-element 0.0))

(gemm! 1.0 W2 z1 0.0 z2)
(axpy! 1.0 b2 z2)
(sigmoid! z2) ;; #<MAT 2x1 BF #2A((0.65839726) (0.8225947))>

(defparameter W3 (make-mat '(2 2) :initial-contents '((0.1 0.2) (0.3 0.4))))
(defparameter b3 (make-mat '(2 1) :initial-contents '((0.1) (0.2))))
(defparameter z3 (make-mat '(2 1) :initial-element 0.0))

(gemm! 1.0 W3 z2 0.0 z3)
(axpy! 1.0 b3 z3) ;; => #<MAT 2x1 F #2A((0.3168271) (0.6962791))>

(time
 (progn
   (gemm! 1.0 W1 x 0.0 z1)
   (axpy! 1.0 b1 z1)
   (sigmoid! z1)
   (gemm! 1.0 W2 z1 0.0 z2)
   (axpy! 1.0 b2 z2)
   (sigmoid! z2)
   (gemm! 1.0 W3 z2 0.0 z3)
   (axpy! 1.0 b3 z3)))
