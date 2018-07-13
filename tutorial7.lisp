;; (ql:quickload :mgl)

(defpackage mgl-learning
  (:use :mgl :cl)
  (:nicknames :m-learning))

(in-package :m-learning)

(setf *default-mat-ctype* :float)
(setf *cuda-enabled* t)

(defparameter *dict-size* 100000)
(defparameter *n-inputs* *dict-size*)
(defparameter *n-hiddens* (* 4 *n-inputs*))

(defclass encoder (fnn)
  ())

(defclass decoder (fnn)
  ())

(defun make-encoder (&key (n-hiddens *n-hiddens*))
  (build-rnn ()
    (build-fnn (:class 'encoder)
      (input (->input :size *n-inputs*))
      (h (->lstm input :name 'h :size n-hiddens)))))

(defun make-decoder (&key (n-hiddens *n-hiddens*))
  (build-rnn ()
    (build-fnn (:class 'decoder)
      (input (->input :size *n-inputs*))
      (h (->lstm input :name 'h :size n-hiddens))
      (prediction (->softmax-xe-loss
                   (->activation h :name 'prediction
                                   :size *n-inputs*))))))

