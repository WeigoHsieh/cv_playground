(function () {


    const fileUploadEl = document.getElementById('file-upload'),
        srcImgEl = document.getElementById('src-image')


    const fileDomProcess = function () {
        fileUploadEl.addEventListener("change", function (e) {
            srcImgEl.src = URL.createObjectURL(e.target.files[0]);
        }, false);
    }

    const GaussianBlur = (src,dst)=> {
        let ksize = new cv.Size(3, 3)
        cv.GaussianBlur(src, dst, ksize, 0, 0, cv.BORDER_DEFAULT);
    }

    const blur = function (src, dst) {
        let anchor = new cv.Point(-1, -1);
        let ksize = new cv.Size(3, 3)
        cv.blur(src, dst, ksize, anchor, cv.BORD_DEFAULT)
    }

    // const findContours = function (dst) {
    //     cv.findContours (dst,)
    // }

    const canny = (src, dst) => {
        cv.Canny(src, dst, 75, 120, 3, false)
    }


    //HoughCircles(image, method, dp, minDist[, circles[, param1[, param2[, minRadius[, maxRadius]]]]]) -> circles
    const houghCircles = function (src, dst) {
        cv.HoughCircles(src, dst, cv.HOUGH_GRADIENT, 1, 45, 75, 40, 0, 0);
    }


    const preImg = function () {
        let src = cv.imread(srcImgEl);
        const dst = new cv.Mat();
        let ksize = new cv.Size(3, 3)
        // const thresh = cv.threshold(src,dst, 170, 255, cv.THRESH_BINARY);
        // cv.cvtColor(src, src, cv.COLOR_RGB2GRAY, 0);
        canny(src, dst)
        // houghCircles(src, dst, cv.HOUGH_GRADIENT, 1, 45, 75, 40, 0, 0);
        // GaussianBlur (src,dst)
        // cv.adaptiveThreshold(src, dst, 200, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 3, 2);
        
        cv.imshow('the-canvas', dst);

        src.delete();
        dst.delete();
    }

    const simpleBlobDetector = function (image, params) {
        params = {}
    }


    // const videoCaptureOnLoad = function () {
    //     const cap = cv.VideoCapture(0).then((e)=>{
    //         console.log(e)
    //     })
    //     if(cap.isOpened()) console.log('opended')
    // }

    const start = function () {
        //檔案上傳處理
        fileDomProcess()

        //預處理
        srcImgEl.onload = preImg

        //CV load
        window.onOpenCvReady = function () {
            // //影像處理
            // videoCaptureOnLoad()
            document.getElementById('loading-opencv-msg').remove();
        }
    }
    start()

})()




// const fileDomProcess = function () {
//     fileUploadEl.addEventListener("change", function (e) {
//         srcImgEl.src = URL.createObjectURL(e.target.files[0]);
//     }, false);
// }

