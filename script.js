let IMAGE_PATH = 'image.jpeg';
let input_img,
    output_img,
    model,
    adj_tensor,
    input_Canvas,
    output_Canvas,
    filterSelector,
    canvasWidth,
    canvasHeight,
    data,
    file;

function preload() {
    canvasWidth = 400;
    canvasHeight = 400;

    model = tf.sequential();

    tf.tidy(() => {
        model.add(
            tf.layers.conv2d({
                inputShape: [canvasHeight, canvasWidth, 3],
                kernelSize: 1,
                filters: 3,
                strides: 1,
                padding: 'same',
                activation: 'relu',
                kernelInitializer: 'varianceScaling',
            })
        );

        //outputlayer
        const output = tf.layers.dense({
            units: 3,
            activation: 'softmax',
        });

        model.add(output);
        const optimizer = tf.train.adam();
        model.compile({
            optimizer: optimizer,
            loss: tf.losses.meanSquaredError,
        });

        model.summary();
    });
}

// async function imageProcessing() {
//     image(img, 0, 0, width, height);
//     img_tensor = tf.browser.fromPixels(c.canvas);
//     clear();

//     switch (filterSelector.value()) {
//         case 'invert':
//             adj_tensor = tf.tidy(() => {
//                 return tf.scalar(255).sub(img_tensor).div(255);
//             });
//             break;
//         case 'cos':
//             adj_tensor = tf.tidy(() => {
//                 return img_tensor.mul([1, 1, 1]).cos().abs().clipByValue(0, 1);
//             });
//             break;
//         case 'darken':
//             adj_tensor = tf.tidy(() => {
//                 return img_tensor.sub(128).div(255).clipByValue(0, 1);
//             });
//             break;
//         case 'lighten':
//             adj_tensor = tf.tidy(() => {
//                 return img_tensor.add(128).div(255).clipByValue(0, 1);
//             });
//             break;
//         case 'clip':
//             adj_tensor = tf.tidy(() => {
//                 return img_tensor.clipByValue(0, 200).div(255);
//             });
//             break;
//         case 'sqrt':
//             adj_tensor = tf.tidy(() => {
//                 return tf.cast(img_tensor, 'float32').rsqrt().clipByValue(0, 1);
//             });
//             break;
//         case 'gray':
//             let arr = await img_tensor.array();
//             for (let i = 0; i < arr.length; i++) {
//                 for (let j = 0; j < arr[i].length; j++) {
//                     let r = arr[i][j][0];
//                     let g = arr[i][j][1];
//                     let b = arr[i][j][2];
//                     let gray = (r + g + b) / 3;
//                     arr[i][j][0] = arr[i][j][1] = arr[i][j][2] = gray;
//                 }
//             }
//             adj_tensor = tf.tidy(() => tf.tensor3d(arr).div(255));
//             break;
//         case 'log':
//             adj_tensor = tf.tidy(() => {
//                 return tf
//                     .cast(img_tensor, 'float32')
//                     .log()
//                     .clipByValue(0, 5)
//                     .div(5);
//             });
//             break;
//         case 'sharpen':
//             adj_tensor = tf.tidy(() => {
//                 return tf
//                     .avgPool(tf.cast(img_tensor, 'float32'), [3, 1], 1, 'same')
//                     .clipByValue(0, 255)
//                     .div(255);
//             });
//             break;
//         case 'else':
//             adj_tensor = tf.tidy(() => {
//                 return img_tensor;
//             });
//             break;
//         default:
//             adj_tensor = img_tensor;
//             break;
//     }

//     tf.browser.toPixels(adj_tensor, c.canvas);
//     adj_tensor.dispose();
//     img_tensor.dispose();

//     // output_tensor = model.predict(img_tensor);
// }

function fitModel() {
    let img_tensor = tf.browser.fromPixels(input_Canvas.canvas);
    img_tensor = img_tensor.reshape([1, canvasHeight, canvasWidth, 3]).div(255);

    let output_tensor = tf.browser.fromPixels(output_Canvas.elt);
    output_tensor = output_tensor.reshape([1, canvasHeight, canvasWidth, 3]);

    model.fit(img_tensor, output_tensor, {
        epochs: 3,
        callbacks: {
            onEpochEnd: (epoch, logs) => {
                console.log(epoch, nf(logs.loss, 2, 3));
                // generateOutputImage();
            },
        },
    });
}

function generateOutputImage() {
    tf.tidy(() => {
        let img_tensor = tf.browser.fromPixels(input_Canvas.canvas);
        img_tensor = img_tensor
            .reshape([1, canvasHeight, canvasWidth, 3])
            .div(255);

        let output_tensor = model.predict(img_tensor);
        output_tensor = output_tensor.reshape([canvasHeight, canvasWidth, 3]);

        // output_tensor.print();

        //clear output canvas
        output_Canvas.elt
            .getContext('2d')
            .clearRect(0, 0, canvasWidth, canvasHeight);

        tf.browser.toPixels(output_tensor, output_Canvas.elt);
    });
}

function setup() {
    // create input canvas
    input_Canvas = createCanvas(canvasWidth, canvasHeight);
    input_Canvas.style('border', '1px solid black');

    // create output canvas
    output_Canvas = createElement('canvas');
    output_Canvas.attribute('width', canvasWidth);
    output_Canvas.attribute('height', canvasHeight);
    output_Canvas.style('border', '1px solid black');

    // load image from file as input
    x = createElement('input');
    x.attribute('type', 'file');
    x.attribute('accept', 'image/*');
    x.style('font-size', '24px');
    x.changed(() => {
        let fl = new FileReader();
        fl.onload = function (e) {
            let path = e.target.result;
            input_Canvas.clear();
            input_img = createImg(path, 'image', 'anonymous', () => {
                image(input_img, 0, 0, width, height);
            });
            input_img.hide();

            x.elt.value = '';
        };
        fl.readAsDataURL(x.elt.files[0]);
    });

    // load image from file as output
    y = createElement('input');
    y.attribute('type', 'file');
    y.attribute('accept', 'image/*');
    y.style('font-size', '24px');
    y.changed(() => {
        let fl = new FileReader();
        fl.onload = function (e) {
            let path = e.target.result;

            //clear output canvas
            output_Canvas.elt
                .getContext('2d')
                .clearRect(0, 0, canvasWidth, canvasHeight);

            output_img = createImg(path, 'image', 'anonymous', () => {
                output_Canvas.elt
                    .getContext('2d')
                    .drawImage(output_img.elt, 0, 0, width, height);
            });
            output_img.hide();

            y.elt.value = '';
        };
        fl.readAsDataURL(y.elt.files[0]);
    });

    // create button to generate output
    button = createButton('generate');
    button.style('font-size', '24px');
    button.mousePressed(generateOutputImage);

    // create button to fit model
    button = createButton('fit');
    button.style('font-size', '24px');
    button.mousePressed(fitModel);

    // filterSelector = createSelect();
    // filterSelector.style('font-size', '24px');
    // filterSelector.position(10, 10);
    // filterSelector.option('normal');
    // filterSelector.option('darken');
    // filterSelector.option('lighten');
    // filterSelector.option('sharpen');
    // filterSelector.option('invert');
    // filterSelector.option('clip');
    // filterSelector.option('cos');
    // filterSelector.option('sqrt');
    // filterSelector.option('gray');
    // filterSelector.option('log');
    // filterSelector.option('else');
    // filterSelector.selected('normal');
    // filterSelector.changed(imageProcessing);
    // img = createImg(IMAGE_PATH, 'image', 'anonymous', imageProcessing);

    noLoop();
}
