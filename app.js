(function () {
    const fileUploadEl = document.getElementById('file-upload'),
        srcImgEl = document.getElementById('src-image')


    const fileDomProcess = function () {
        fileUploadEl.addEventListener("change", function (e) {
            srcImgEl.src = URL.createObjectURL(e.target.files[0]);
        }, false);
    }
    const preImg = function () {
        const src = cv.imread(srcImgEl); // load the image from <img>
        const dst = new cv.Mat();

        cv.cvtColor(src, src, cv.COLOR_RGB2GRAY, 0);

        cv.Canny(src, dst, 50, 100, 3, false); // You can try more different parameters
        cv.imshow('the-canvas', dst); // display the output to canvas

        src.delete(); // remember to free the memory
        dst.delete();
    }

    const start = function () {
        fileDomProcess()

        srcImgEl.onload = preImg

        // opencv loaded?
        window.onOpenCvReady = function () {
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

