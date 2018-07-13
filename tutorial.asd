(asdf:defsystem "MGL-tutorial"
  :description "mgl tutorial"
  :version "0.0.1"
  :author "Meguru Mokke"
  :licence "MIT"
  :depends-on (:cl :mgl-mat :alexandria
                   :metabang-bind
                   :cl-libsvm-format)
  :serial t
  :components ((:file "tutorial.lisp")
               ;;(:file "tutorial2.lisp")
               ;;(:file "tutorial3.lisp")
               ))
