(ql:quickload '(:mgl-mat
                :metabang-bind
                :cl-libsvm-format))

(defpackage cl-zerodl
  (:use :cl :mgl-mat :metabang-bind)
  (:nicknames :zerodl))

(in-package :zerodl)

(defmacro define-class
    (class-name superclass-list &body body)
  `(defclass ,class-name (,@superclass-list)
     ,(mapcar (lambda (slot)
                 (let* ((slot-symbol (if (listp slot) (car slot) slot))
                        (slot-name (symbol-name slot-symbol))
                        (slot-initval (if (listp slot) (cadr slot) nil)))
                   (list slot-symbol
                         :accessor (intern slot-name)
                         :initarg (intern slot-name :keyword)
                         :initform slot-initval)))
       body)))

;; (intern "hoge") => |hoge|
;; (intern "hoge" :keyword) => :|hoge|

(setf *default-mat-ctype* :float
      *cuda-enabled* t
      *print-mat* t
      *print-length* 100
      *print-level* 10)

(define-class layer ()
  input-dimentions output-dimensions
  forward-out backward-out)

(defgeneric forward (layer &rest inputs))
(defgeneric backward (layer dout))

;;; multiple layer -------------------------------------------
(define-class multiple-layer (layer) x y)

(defun make-multiple-layer (input-dimentions)
  (make-instance 'multiple-layer
                 :input-dimentions input-dimentions
                 :output-dimensions input-dimentions
                 :forward-out (make-mat input-dimentions)
                 :backward-out (list (make-mat input-dimentions) ;; dx
                                     (make-mat input-dimentions));; dy
                 :x (make-mat input-dimentions)
                 :y (make-mat input-dimentions)))

;; geem => elements' mult <=> gemm => matrixs' mult
(defmethod forward ((layer multiple-layer) &rest inputs)
  (bind ((out (forward-out layer))
         ((x y) inputs))
    (copy! x (x layer)) ;; x -> layer's x
    (copy! y (y layer))
    (geem! 1.0 x y 0.0 out))) ;; x .* y => out

;; dout means the backword's input , x/y means stored parameter for the forward.
(defmethod backward ((layer multiple-layer) dout)
  (let* ((out (backward-out layer))
         (dx (car  out))
         (dy (cadr out)))
    (geem! 1.0 dout (y layer) 0.0 dx)
    (geem! 1.0 dout (x layer) 0.0 dy)
    out))
;;; ---------------------------------------------------------------

;;; add layer -----------------------------------------------------
(define-class add-layer (layer))

(defun make-add-layer (input-dimentions)
  (make-instance 'add-layer
                 :input-dimentions input-dimentions
                 :output-dimensions input-dimentions
                 :forward-out (make-mat input-dimentions)
                 :backward-out (list (make-mat input-dimentions)   ; dx
                                     (make-mat input-dimentions)))); dy

(defmethod forward ((layer add-layer) &rest inputs) ;; input (a . b)
  (let ((out (forward-out layer)))
    (copy! (car inputs) out)
    (axpy! 1.0 (cadr inputs) out))) ;; a + b => out

(defmethod backward ((layer add-layer) dout)
  (bind ((out (backward-out layer))
         ((dx dy) out))
    (copy! dout dx) ;; dout => dx
    (copy! dout dy)
    out))
;; ----------------------------------------------------------------

;;;

;; inputs
(defparameter n-apple  (make-mat '(1 1) :initial-element 2.0))
(defparameter apple    (make-mat '(1 1) :initial-element 100.0))
(defparameter orange   (make-mat '(1 1) :initial-element 150.0))
(defparameter n-orange (make-mat '(1 1) :initial-element 3.0))
(defparameter tax      (make-mat '(1 1) :initial-element 1.1))

;; layers
(defparameter mul-apple-layer
  (make-multiple-layer '(1 1)))
(defparameter mul-orange-layer
  (make-multiple-layer '(1 1)))
(defparameter add-apple-orange-layer
  (make-add-layer '(1 1)))
(defparameter mul-tax-layer
  (make-multiple-layer '(1 1)))

;; forward
(let* ((apple-price  (forward mul-apple-layer apple n-apple))
       (orange-price (forward mul-orange-layer orange n-orange))
       (all-price    (forward add-apple-orange-layer apple-price orange-price))
       (price    (forward mul-tax-layer all-price tax)))
  (print price))

(bind ((dprice (make-mat '(1 1) :initial-element 1.0))
       ((dall-price dtax) (backward mul-tax-layer dprice))
       ((dapple-price dorange-price) (backward add-apple-orange-layer dall-price))
       ((dorange dnorange) (backward mul-orange-layer dorange-price))
       ((dapple dnapple) (backward mul-apple-layer dapple-price))
       )
  (print (list dnapple dapple dorange dnorange dtax)))
;; => (#<MAT 1x1 AB #2A((110.0))> #<MAT 1x1 AB #2A((2.2))>
;;     #<MAT 1x1 AB #2A((3.3000002))> #<MAT 1x1 AB #2A((165.0))>
;;     #<MAT 1x1 AB #2A((650.0))>)

;;; relu layer ---------------------------------------------------------------
(define-class relu-layer (layer)
  zero mask)

(defun make-relu-layer (input-dimentions)
  (make-instance 'relu-layer
                 :input-dimentions input-dimentions
                 :output-dimensions input-dimentions
                 :forward-out (make-mat input-dimentions)
                 :backward-out (make-mat input-dimentions)
                 :zero (make-mat input-dimentions :initial-element 0.0)
                 :mask (make-mat input-dimentions :initial-element 0.0)))

(defmethod forward ((layer relu-layer) &rest inputs)
  (let ((zero (zero layer))
        (mask (mask layer))
        (out  (forward-out layer)))
    ;; set mask
    (copy! (car inputs) mask)
    (.<! zero mask) ;; zero < mask => 1 : otherwise => 1
    ;; set output
    (copy! (car inputs) out)
    (.max! 0.0 out)))

(defmethod backward ((layer relu-layer) dout)
  (geem! 1.0 dout (mask layer) 0.0 (backward-out layer))) ;; dout .* mask => backward-out

;; -------------------------------------------------------------------

;; sigmoid layer -----------------------------------------------------
(define-class sigmoid-layer (layer))

(defun make-sigmoid-layer (input-dimentions)
  (make-instance 'sigmoid-layer
                 :input-dimentions input-dimentions
                 :output-dimensions input-dimentions
                 :forward-out (make-mat input-dimentions)
                 :backward-out (make-mat input-dimentions)))

(defmethod forward ((layer sigmoid-layer) &rest inputs)
  (let ((out (forward-out layer)))
    (copy! (car inputs) out)
    (.logistic! out)))

(defmethod backward ((layer sigmoid-layer) dout)
  (let ((y (forward-out layer))
        (out (backward-out layer)))
    (copy! y out)
    (.+! -1.0 out) ;; -1 .* y
    (geem! -1.0 y out 0.0 out) ;; -y .* (-1 + y)
    (.*! dout out))) ;; dout .* -y .* (-1 + y)
;; -------------------------------------------------------------------

;; affine-layer ------------------------------------------------------
;; affine = dot
(define-class affine-layer (layer)
  x weight bias)

;; x : (batch-size, in-size)
;; y : (batch-size, out-size)
;; W : (in-size, out-size)
;; b : (out-size)

(defun make-affine-layer (input-dimensions output-dimensions)
  (let ((weight-dimensions (list (cadr input-dimensions) (cadr output-dimensions)))
        (bias-dimension (cadr output-dimensions)))
    (make-instance 'affine-layer
                   :input-dimentions input-dimensions
                   :output-dimensions output-dimensions
                   :forward-out (make-mat output-dimensions)
                   :backward-out (list (make-mat input-dimensions) ;; dx
                                       (make-mat weight-dimensions);; dW
                                       (make-mat bias-dimension)) ;; db
                   :x (make-mat input-dimensions)
                   :weight (make-mat weight-dimensions)
                   :bias (make-mat bias-dimension))))

(defmethod forward ((layer affine-layer) &rest inputs)
  (let* ((x (car inputs))
         (W (weight layer))
         (b (bias layer))
         (out (forward-out layer)))
    (copy! x (x layer))
    (fill! 1.0 out)
    (scale-columns! b out) ;; for i in out's column b * out_i
    (gemm! 1.0 x W 1.0 out))) ;; x * W + out => out

(defmethod backward ((layer affine-layer) dout)
  (bind (((dx dW db) (backward-out layer)))
    (gemm! 1.0 dout (weight layer) 0.0 dx :transpose-b? t) ;; dout * weight^T => dL/dx
    (gemm! 1.0 (x layer) dout 0.0 dW :transpose-a? t)   ;; x^T * dout => dL/dW : dout = dL/dY
    (sum! dout db :axis 0)
    (backward-out layer)))
;; -------------------------------------------------------------------

;; softmax -------------------------------------------
(defun average! (a batch-size-tmp)
  (sum! a batch-size-tmp :axis 1)
  (scal! (/ 1.0 (mat-dimension a 1)) batch-size-tmp))

(defun softmax! (a result batch-size-tmp &key (avoid-overflow-p t))
  ;; in order to avoid overflow, subtract average value for each column
  (when avoid-overflow-p
    (average! a batch-size-tmp)
    (fill! 1.0 result)
    (scale-rows! batch-size-tmp result)
    (axpy! -1.0 result a)) ;; a - average(a)
  (.exp! a) ;; exp (a - average(a))
  (sum! a batch-size-tmp :axis 1) ;; sum (exp(...) + average(a))
  (fill! 1.0 result)
  (scale-rows! batch-size-tmp result)
  (.inv! result) ;; 1/.
  (.*! a result)) ;; a * 1/.

;; cross-entropy! ----------------------------------------------------
(defun cross-entropy! (y target tmp  batch-size-tmp &key (delta 1e-7))
  (let ((batch-size (mat-dimension target 0)))
    (copy! y tmp)
    (.+! delta tmp) ;; log 0 -> - \inf
    (.log! tmp)
    (.*! target tmp) ;; target * log.
    (sum! tmp batch-size-tmp :axis 1) ;; Sigma Sigma target * log
    (/ (asum batch-size-tmp) batch-size)))
;; -------------------------------------------------------------------


;; softmax/loss layer ------------------------------------------------
(define-class softmax/loss-layer (layer)
  loss y target batch-size-tmp)

(defun make-softmax/loss-layer (input-dimensions)
  (make-instance 'softmax/loss-layer
                 :input-dimentions input-dimensions
                 :output-dimensions 1
                 :backward-out (make-mat input-dimensions)
                 :y (make-mat input-dimensions)
                 :target (make-mat input-dimensions)
                 :batch-size-tmp (make-mat (car input-dimensions))))

(defmethod forward ((layer softmax/loss-layer) &rest inputs)
  (bind (((x target) inputs)
         (tmp (target layer))
         (y (y layer))
         (batch-size-tmp (batch-size-tmp layer)))
    (copy! x tmp)
    (softmax! tmp y batch-size-tmp) ;; softmax x => y
    (let ((out (cross-entropy! y target tmp batch-size-tmp))) ;; cross-entropy y target
      (copy! target (target layer))
      (setf (forward-out layer) out)
      out)))

(defmethod backward ((layer softmax/loss-layer) dout)
  (let* ((target (target layer))
         (y (y layer))
         (out (backward-out layer))
         (batch-size (mat-dimension target 0))) ;; mat '(3 2) => 3
    (copy! y out)
    (axpy! -1.0 target out)
    (scal! (/ 1.0 batch-size) out)))
;; -----------------------------------------------------------------
(define-class network ()
  layers last-layer batch-size)

(defun make-network (input-size hidden-size output-size batch-size
                     &key (weight-init-std 0.01))
  (let* ((network
           (make-instance 'network
                          :layers (vector
                                   (make-affine-layer (list batch-size input-size)
                                                      (list batch-size hidden-size))
                                   (make-relu-layer (list batch-size hidden-size))
                                   (make-affine-layer (list batch-size hidden-size)
                                                      (list batch-size output-size)))
                          :last-layer (make-softmax/loss-layer (list batch-size output-size))
                          :batch-size batch-size))
         (W1 (weight (svref (layers network) 0)))
         (W2 (weight (svref (layers network) 2))))
    ;; initialize W1 W2
    (gaussian-random! W1)
    (scal! weight-init-std W1)
    (gaussian-random! W2)
    (scal! weight-init-std W2)
    network))

(defun predict (network x)
  (loop for layer across (layers network) do
    (setf x (forward layer x)))
  x)

(defun network-loss (network x target)
  (let ((y (predict network x)))
    (forward (last-layer network) y target)))

(defun set-gradient! (network x target)
  (let ((layers (layers network))
        dout)
    ;; forward
    (network-loss network x target)
    ;; backward
    (setf dout (backward (last-layer network) 1.0))
    (loop for i from (1- (length layers)) downto 0 do
      (let ((layer (svref layers i)))
        (setf dout (backward layer (if (listp dout) (car dout) dout)))))))

(defmacro do-index-value-list ((index value list) &body body)
  (let ((iter (gensym))
        (inner-list (gensym)))
    `(labels ((,iter (,inner-list)
                (when ,inner-list
                  (let ((,index (car ,inner-list))
                        (,value (cadr ,inner-list)))
                    ,@body)
                  (,iter (cddr ,inner-list)))))
       (,iter ,list))))
;; function iter
;; args : inner-list
;; when inner-list isn't nil -> body
;;                 otherwise -> (iter cddr inner-list)

(defun read-data (data-path data-dimension n-class &key (most-min-class 1))
  (let* ((data-list (svmformat:parse-file data-path))
         (len (length data-list))
         (target (make-array (list len n-class)
                             :element-type 'single-float
                             :initial-element 0.0))
         (datamatrix (make-array (list len data-dimension)
                                 :element-type 'single-float
                                 :initial-element 0.0)))
    (loop for i fixnum from 0
          for datum in data-list
          do (setf (aref target i (- (car datum) most-min-class)) 1.0)
             (do-index-value-list (j v (cdr datum))
               (setf (aref datamatrix i (- j most-min-class)) v)))
    (values (array-to-mat datamatrix) (array-to-mat target))))

;; --------------------------------------------------------
;; load mnist
(multiple-value-bind (datamat target)
    (read-data "./mnist.scale" 784 10 :most-min-class 0)
  (defparameter mnist-dataset datamat)
  (defparameter mnist-target target))


(multiple-value-bind (datamat target)
    (read-data "./mnist.scale.t" 784 10 :most-min-class 0)
  (defparameter mnist-dataset-test datamat)
  (defparameter mnist-target-test target))

(setf *print-mat* nil)

;; mnist-dataset #<MAT 60000x784 AB {1019D7E323}>

(defun set-mini-batch! (dataset start-row-index batch-size)
  (let ((dim (mat-dimension dataset 1)))
    (reshape-and-displace! dataset
                           (list batch-size dim)
                           (* start-row-index dim))))

(defun reset-shape! (dataset)
  (let* ((dim (mat-dimension dataset 1))
         (len (/ (mat-max-size dataset) dim)))
    (reshape-and-displace! dataset (list len dim) 0)))


(set-mini-batch! mnist-dataset 0 100) ;; look 0 to 100
;; #<MAT 0+100x784+46961600 AB {1019D7E323}>

(reset-shape! mnist-dataset) ;; reset
;; #<MAT 60000x784 AB {1019D7E323}>

;; train -------------------------------------------------------------
(defun train (network x target &key (learning-rate 0.1))
  (set-gradient! network x target)
  (bind ((layer (aref (layers network) 0))
         ((dx dW dB) (backward-out layer)))
    (declare (ignore dx))
    (axpy! (- learning-rate) dW (weight layer))
    (axpy! (- learning-rate) dB (bias layer)))
  (bind ((layer (aref (layers network) 2))
         ((dx dW dB) (backward-out layer)))
    (declare (ignore dx))
    (axpy! (- learning-rate) dW (weight layer)) ;; - learning-rate * dW + W => W
    (axpy! (- learning-rate) dB (bias layer)))) ;; - learning-rate * dB + B => bias

(defparameter mnist-network (make-network 784 50 10 100))

;; (time
;;  (loop repeat 10000 do
;;    (let* ((batch-size (batch-size mnist-network))
;;           (rand (random (- 60000 batch-size))))
;;      (set-mini-batch! mnist-dataset rand batch-size)
;;      (set-mini-batch! mnist-target rand batch-size)
;;      (train mnist-network mnist-dataset mnist-target))))
;; =>
;; Evaluation took:
;; 81.566 seconds of real time
;; 81.537878 seconds of total run time (81.535430 user, 0.002448 system)
;; [ Run times consist of 0.011 seconds GC time, and 81.527 seconds non-GC time. ]
;; 99.97% CPU
;; 277,980,197,958 processor cycles
;; 371,827,504 bytes consed

(defparameter *let-output-through-p* t)


(defun max-position-column (arr)
  (declare (optimize (speed 3) (space 0) (safety 0) (debug 0))
           (type (array single-float) arr))
  (let ((max-arr (make-array (array-dimension arr 0)
                             :element-type 'single-float
                             :initial-element most-negative-single-float))
        (pos-arr (make-array (array-dimension arr 0)
                             :element-type 'fixnum
                             :initial-element 0)))
    (loop for i fixnum from 0 below (array-dimension arr 0) do
      (loop for j fixnum from 0 below (array-dimension arr 1) do
        (when (> (aref arr i j) (aref max-arr i))
          (setf (aref max-arr i) (aref arr i j)
                (aref pos-arr i) j))))
    pos-arr))

(defun predict-class (network x)
  (max-position-column (mat-to-array (predict network x))))

(defun accuracy (network dataset target)
  (let* ((batch-size (batch-size network))
         (dim (mat-dimension dataset 1))
         (len (/ (mat-max-size dataset) dim))
         (cnt 0))
    (loop for n from 0 to (- len batch-size) by batch-size do
      (set-mini-batch! dataset n batch-size)
      (set-mini-batch! target n batch-size)
      (incf cnt
            (loop for pred across (predict-class network dataset)
                  for tgt  across (max-position-column (mat-to-array target))
                  count (= pred tgt))))
    (* (/ cnt len) 1.0)))

;; (predict-class mnist-network mnist-dataset)

;; train and test
(defparameter minist-network (make-network 784 256 10 100))
(defparameter train-acc-list nil)
(defparameter test-acc-list nil)

(loop for i from 1 to 100000 do
  (let* ((batch-size (batch-size mnist-network))
         (rand (random (- 60000 batch-size))))
    (set-mini-batch! mnist-dataset rand batch-size)
    (set-mini-batch! mnist-target rand batch-size)
    (train mnist-network mnist-dataset mnist-target)
    (when (zerop (mod i 600))
      (let ((train-acc (accuracy mnist-network mnist-dataset mnist-target))
            (test-acc (accuracy mnist-network mnist-dataset-test mnist-dataset-test)))
        (format t "cycle: ~A~,15Ttrain-acc: ~A~,10Ttest-acc: ~A~%" i train-acc test-acc)
        (push train-acc train-acc-list)
        (push test-acc test-acc-list)))))
