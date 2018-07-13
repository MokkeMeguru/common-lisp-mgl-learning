;; (ql:quickload :mgl)
;; (asdf:load-system :mgl)

(defpackage rnn-learning
  (:use :mgl :cl)
  (:nicknames :r-learning))

(in-package :r-learning)

;; 各ループにおける入力は一つです。
(defparameter *n-inputs* 1)

;; 学習したい内容は、これまでの入力の総和の符号を出力するという規則です。
(defparameter *n-outputs* 3)

;; ランダムな値を持つ長さ 1 から LENGTH の教師データを作成します。
;; 一つの教師データに含まれている要素は以下の２つです。
;; 1. ネットワークに対する入力 (一つのランダムな数字)
;; 2. 0, 1, 2 で構成される合計の値の符号 (0 = negative, 1 = zero, 2 = positive)
;;    ひねりを加えるために、入力が負であった場合にはそれまでの合計がリセットされます。
(defun make-sum-sign-instance (&key (length 10))
  (let ((length (max 1 (random length)))
        (sum 0))
    (loop for i below length
          collect
          (let ((x (1- (* 2 (random 2))))) ;; 1 か -1
            (incf sum x)
            (when (< x 0)
              (setq sum x)) ;; 合計のリセット
            (list x (cond ((minusp sum) 0) ;; negative
                          ((zerop sum) 1) ;; zero
                          (t 2))))))) ;; positive

;; example:
;; list ((1 2) (1 2) (-1 0)    (-1 0)   (1 1) (1 2) (-1 0)   (-1 0))
;; sum     1     1    -1(reset) -1(reset) 0     1     0(reset) 0(reset)

;; LSTMの隠れ層を一つ持ち、softmax関数を用いた出力層を持ったRNNを作成します。
;; 各ループに対して、sum-sign-fnn はインスタンス化されます。
(defun make-sum-sign-rnn (&key (n-hiddens 1))
  (build-rnn ()
    (build-fnn (:class 'sum-sign-fnn)
      (input (->input :size 1)) ;; input layer
      (h (->lstm input :name 'h :size n-hiddens)) ;; hidden layer
      (prediction (->softmax-xe-loss (->activation h :name 'prediction ;; output layer
                                                     :size *n-outputs*))))))

;; [layer]      input   ->    hidden    ->   output
;; [function] '#->input     '#->lstm        '#softmax
;; [size]         1             1               3

;; note : ACTIVATION で何をやっているか
;;   sum_i(x_i * W_(i)) + sum_j(y_j .* V_j) + biases
;;     where x_i : input lump 入力
;;           W_i : x_i に対する 重み密行列
;;           V_j : y_i に対して要素的に乗算される peephole の接続ベクトル
;;           y_j : 入力

;; 入力を変換する set-input 関数を後付するために fnn を継承したクラスを定義します。
(defclass sum-sign-fnn (fnn)
  ())

;; この RNN に与えるサンプルを作成する make-sum-sign-instance からインスタンスのバッヂを得る関数
;; 同じタイムステップに属するこれらのインスタンスの要素で呼び出され、その入力と目標のリストを作成します。
(defmethod set-input (instances (fnn sum-sign-fnn))
  (let ((input-nodes (nodes (find-clump 'input fnn))))
    (setf (target (find-clump 'prediction fnn))
          (loop for stripe upfrom 0
                for instance in instances
                collect
                ;; バッヂ内のそれぞれのシーケンスの長さは等しいとは限りません
                ;; シーケンスを読みきった場合、RNN は nil を送ります。
                (when instance
                  (destructuring-bind (input target) instance
                    (setf (mref input-nodes stripe 0) input)
                    target))
                ;; input-nodes : [input_1  input_2  ... input_n  nil ... nil]
                ;; target      : [target_1 target_2 ... target_n nil ... nil]
                ))))

;; max-n=samples 個のランダムな値を作成するオブジェクトを作ります。
(defun make-sampler (max-n-samples &key (length 10))
  (make-instance 'function-sampler :max-n-samples max-n-samples
                 :generator (lambda () (make-sum-sign-instance :length length))))

;; クロスエントロピーを用いた損失を最小化させるようにネットワークをトレーニングします。
;; ここでは Adam optimizer を用いています。
(defun train-sum-sign-rnn ()
  (let ((rnn (make-sum-sign-rnn)))
    ;; バッヂをいくつのサブバッヂに分割するかを指定します。
    (setf (max-n-stripes rnn) 50)
    ;; 重みを sqrt(1 / fan-in) で初期化します。
    ;; fan-in = (mat-dimension (nodes weight) 0)
    (map-segments (lambda (weights)
                    (let* ((fan-in (mat-dimension (nodes weights) 0))
                           (limit (sqrt (/ 6 fan-in))))
                      (uniform-random! (nodes weights)
                                       :limit (* 2 limit))
                      (.+! (- limit) (nodes weights))))
                  rnn)
    (minimize (monitor-optimization-periodically
               (make-instance 'adam-optimizer
                              :learning-rate 0.2
                              :mean-decay 0.9
                              :mean-decay-decay 0.9
                              :variance-decay 0.9
                              :batch-size 100)
               '((:fn log-test-error :period 30000)
                 (:fn reset-optimization-monitors :period 30000)))
              (make-instance 'bp-learner
                             :bpn rnn
                             :monitors (make-cost-monitors rnn))
              :dataset (make-sampler 30000))))

;; note : '#node とは
;; The values computed by the lump in the forward
;;          pass are stored here. It is an `N-STRIPES * SIZE` matrix that has
;;          storage allocated for `MAX-N-STRIPES * SIZE` elements for
;;          non-weight lumps. ->WEIGHT lumps have no stripes nor restrictions
;;          on their shape.
;; フォワードアルゴリズムで計算された値がここに格納されています。
;; 格納されているそれは、'max-n-stripes * size' の要素に割り当てられた記憶域を持った
;;  n-stripe * size 行列です。

;; テストの誤差を記録する関数です。また、学習の最初の optimizer と bpn についての情報も記録します。
(defun log-test-error (optimizer learner)
  (when (zerop (n-instances optimizer))
    (describe optimizer)
    (describe (bpn learner)))
  (let ((rnn (bpn learner)))
    (log-padded
     (append
      (monitor-bpn-results (make-sampler 1000) rnn
                           (make-cost-monitors
                            rnn :attributes '(:event "pred.")))
      ;; 別の手法で同じ結果が得られることがあります。
      ;; 長さ２０までのシーケンスの予測を監視するが、
      ;; メモリを節約するため RNN を不必要に展開しないようにします。
      (let ((*warp-time* t))
        (monitor-bpn-results (make-sampler 1000 :length 20) rnn
                             ;; インスタンスの各バッチのあとに監視をリセットします。
                             (make-cost-monitors
                              rnn :attributes '(:event "warped pred."))))))
    ;; それ以上展開が行われないことを確認しています。
    (assert (<= (length (clumps rnn)) 10)))
  (log-mat-room))

;; 
(defparameter model
 (let (;; バックプロパゲーションを行う際には、単精度浮動小数点で十分です。
        ;; 単精度は早くてメモリ消費が抑えられます。
        (*default-mat-ctype* :float)
        ;; 展開されたネットワークが GPU に収まらないほど長いシーケンスでも RNN が正しく動作できるように、
        ;; GPU メモリ間でデータを移動できるようにします。
        (*cuda-window-start-time* 1)
        (*log-time* nil))
    ;; ランダムな値を生成し送ります。
    (repeatably ()
      ;; cuda を有効化します。
      (with-cuda* ()
        (train-sum-sign-rnn)))))

;; training at n-instances: 0
;; cost: 0.000e+0 (0)
;; #<ADAM-OPTIMIZER {1010C5E203}>
;; GD-OPTIMIZER description:
;;   N-INSTANCES = 0
;;   SEGMENT-SET = #<SEGMENT-SET
;; (#<->WEIGHT (H (H . #1=(:OUTPUT))) :SIZE 1 1/1 :NORM 1.73685>
;;  #<->WEIGHT (H (H . #2=(:CELL))) :SIZE 1 1/1 :NORM 0.31893>
;;  #<->WEIGHT (#3=(H . #2#) #4=(H . #5=(:FORGET)) :PEEPHOLE) :SIZE 1 1/1 :NORM 1.81610>
;;  #<->WEIGHT (H #4#) :SIZE 1 1/1 :NORM 0.21965>
;;  #<->WEIGHT (#3# #6=(H . #7=(:INPUT)) :PEEPHOLE) :SIZE 1 1/1 :NORM 1.74939>
;;  #<->WEIGHT (H #6#) :SIZE 1 1/1 :NORM 0.40377>
;;  #<->WEIGHT (H PREDICTION) :SIZE 3 1/1 :NORM 2.15898>
;;  #<->WEIGHT (:BIAS PREDICTION) :SIZE 3 1/1 :NORM 2.94470>
;;  #<->WEIGHT (#3# #8=(H . #1#) :PEEPHOLE) :SIZE 1 1/1 :NORM 0.97601>
;;  #<->WEIGHT (INPUT #8#) :SIZE 1 1/1 :NORM 0.65261>
;;  #<->WEIGHT (:BIAS #8#) :SIZE 1 1/1 :NORM 0.37653>
;;  #<->WEIGHT (INPUT #3#) :SIZE 1 1/1 :NORM 0.92334>
;;  #<->WEIGHT (:BIAS #3#) :SIZE 1 1/1 :NORM 0.01609>
;;  #<->WEIGHT (INPUT #9=(H . #5#)) :SIZE 1 1/1 :NORM 1.09995>
;;  #<->WEIGHT (:BIAS #9#) :SIZE 1 1/1 :NORM 1.41244>
;;  #<->WEIGHT (INPUT #10=(H . #7#)) :SIZE 1 1/1 :NORM 0.40475>
;;  #<->WEIGHT (:BIAS #10#) :SIZE 1 1/1 :NORM 1.75358>) {1010C5FCD3}>
;;   LEARNING-RATE = 2.00000e-1
;;   MOMENTUM = NONE
;;   MOMENTUM-TYPE = :NONE
;;   WEIGHT-DECAY = 0.00000e+0
;;   WEIGHT-PENALTY = 0.00000e+0
;;   N-AFTER-UPATE-HOOK = 0
;;   BATCH-SIZE = 100

;; BATCH-GD-OPTIMIZER description:
;;   N-BEFORE-UPATE-HOOK = 0

;; ADAM-OPTIMIZER description:
;;   MEAN-DECAY = 9.00000e-1
;;   MEAN-DECAY-DECAY = 9.00000e-1
;;   EFFECTIVE-MEAN-DECAY = 9.00000e-1
;;   VARIANCE-DECAY = 9.00000e-1
;;   VARIANCE-ADJUSTMENT = 1.00000d-7
;; #<RNN {1010A251F3}>
;; BPN description:
;;   CLUMPS = #(#<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4>
;;              #<SUM-SIGN-FNN :STRIPES 1/50 :CLUMPS 4>)
;;   N-STRIPES = 1
;;   MAX-N-STRIPES = 50

;; RNN description:
;;   MAX-LAG = 1
;; pred.        cost: 1.223e+0 (4455.00)
;; warped pred. cost: 1.228e+0 (9476.00)
;; Foreign memory usage:
;; foreign arrays: 378 (used bytes: 92,400)
;; CUDA memory usage:
;; device arrays: 114 (used bytes: 220,892, pooled bytes: 19,200)
;; host arrays: 162 (used bytes: 39,600)
;; host->device copies: 6,166, device->host copies: 4,490

;; training at n-instances: 30000
;; cost: 3.822e-2 (138195.00)
;; pred.        cost: 1.426e-6 (4593.00)
;; warped pred. cost: 1.406e-6 (9717.00)
;; Foreign memory usage:
;; foreign arrays: 432 (used bytes: 105,600)
;; CUDA memory usage:
;; device arrays: 148 (used bytes: 224,428, pooled bytes: 19,200)
;; host arrays: 216 (used bytes: 52,800)
;; host->device copies: 465,822, device->host copies: 371,990
;; (#<->WEIGHT (H (H :OUTPUT)) :SIZE 1 1/1 :NORM 0.10619>
;;  #<->WEIGHT (H (H :CELL)) :SIZE 1 1/1 :NORM 0.94439>
;;  #<->WEIGHT ((H :CELL) (H :FORGET) :PEEPHOLE) :SIZE 1 1/1 :NORM 0.61375>
;;  #<->WEIGHT (H (H :FORGET)) :SIZE 1 1/1 :NORM 0.37961>
;;  #<->WEIGHT ((H :CELL) (H :INPUT) :PEEPHOLE) :SIZE 1 1/1 :NORM 1.17974>
;;  #<->WEIGHT (H (H :INPUT)) :SIZE 1 1/1 :NORM 0.88021>
;;  #<->WEIGHT (H PREDICTION) :SIZE 3 1/1 :NORM 49.93931>
;;  #<->WEIGHT (:BIAS PREDICTION) :SIZE 3 1/1 :NORM 10.98254>
;;  #<->WEIGHT ((H :CELL) (H :OUTPUT) :PEEPHOLE) :SIZE 1 1/1 :NORM 0.68144>
;;  #<->WEIGHT (INPUT (H :OUTPUT)) :SIZE 1 1/1 :NORM 0.65386>
;;  #<->WEIGHT (:BIAS (H :OUTPUT)) :SIZE 1 1/1 :NORM 10.22970>
;;  #<->WEIGHT (INPUT (H :CELL)) :SIZE 1 1/1 :NORM 5.98149>
;;  #<->WEIGHT (:BIAS (H :CELL)) :SIZE 1 1/1 :NORM 0.10678>
;;  #<->WEIGHT (INPUT (H :FORGET)) :SIZE 1 1/1 :NORM 4.46346>
;;  #<->WEIGHT (:BIAS (H :FORGET)) :SIZE 1 1/1 :NORM 1.57175>
;;  #<->WEIGHT (INPUT (H :INPUT)) :SIZE 1 1/1 :NORM 0.36392>
;;  #<->WEIGHT (:BIAS (H :INPUT)) :SIZE 1 1/1 :NORM 8.63785>)
