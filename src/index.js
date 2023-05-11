///////////////////////////////////////

function relu(x) {
  return Math.max(0, x);
}

function softmax(x) {
  let s = 0;
  for (let i = 0; i < x.length; i++) {
    s += x[i];
  }
  for (let i = 0; i < x.length; i++) {
    x[i] /= s;
  }
  return x;
}

function dense(x, w, b, f = relu) {
  let out = [];
  for (let j = 0; j < b.length; j++) {
    let n = b[j];
    for (let i = 0; i < x.length; i++) {
      n += x[i] * w[i][j];
    }
    out.push(f(n));
  }
  return out;
}

function predict(num) {
  let out = [];
  out = dense(num, W1, B1);
  out = dense(out, W2, B2);
  out = dense(out, W3, B3, Math.exp);
  return softmax(out);
}

/////////////////////////////////////////

function imageDataToGrayscale(imgData) {
  let grayscaleImg = [];
  for (let y = 0; y < imgData.height; y++) {
    grayscaleImg[y] = [];
    for (let x = 0; x < imgData.width; x++) {
      let offset = y * 4 * imgData.width + 4 * x;
      let alpha = imgData.data[offset + 3];
      if (alpha == 0) {
        imgData.data[offset] = 255;
        imgData.data[offset + 1] = 255;
        imgData.data[offset + 2] = 255;
      }
      imgData.data[offset + 3] = 255;
      grayscaleImg[y][x] = imgData.data[y * 4 * imgData.width + x * 4 + 0] / 255;
    }
  }
  return grayscaleImg;
}

function reduceImage(img) {
  let arr = new Array(28);
  for (let y = 0; y < 28; y++) {
    arr[y] = new Array(28);
    for (let x = 0; x < 28; x++) {
      let mean = 0;
      for (let v = 0; v < 10; v++) {
        for (let h = 0; h < 10; h++) {
          mean += img[y * 10 + v][x * 10 + h];
        }
      }
      arr[y][x] = 1 - mean / 100;
    }
  }
  return arr;
}

function get_shift(arr) {
  let sum_x = 0, sum_y = 0, n = 0;
  for (let x = 0; x < 28; x++) {
    for (let y = 0; y < 28; y++) {
      if (arr[x][y]) {
        sum_x += x;
        sum_y += y;
        n++;
      }
    }
  }
  if (!n) return [0, 0];
  return [14 - parseInt(sum_x / n), 14 - parseInt(sum_y / n)];
}

function centralize(arr) {
  [dx, dy] = get_shift(arr);
  let new_arr = [...Array(28)].map(_ => [...Array(28)].map(_ => 0));
  for (let x = -Math.min(0, dx); x < 28 - Math.max(0, dx); x++) {
    for (let y = -Math.min(0, dy); y < 28 - Math.max(0, dy); y++) {
      if (arr[x][y]) new_arr[x + dx][y + dy] = arr[x][y];
    }
  }
  return new_arr;
}

function flatten(arr) {
  let new_arr = []
  for (let i = 0; i < 28; i++) {
    for (let j = 0; j < 28; j++) {
      new_arr.push(arr[i][j]);
    }
  }
  return new_arr;
}

/////////////////////////////////////

const CANVAS_SIZE = 280;
const CANVAS_SCALE = 1;

const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const clearButton = document.getElementById('clear-button');

let isMouseDown = false;
let hasIntroText = true;
let lastX = 0;
let lastY = 0;

// Add 'Draw a number here!' to the canvas.
ctx.lineWidth = 28;
ctx.lineJoin = 'round';
ctx.font = '28px sans-serif';
ctx.textAlign = 'center';
ctx.textBaseline = 'middle';
ctx.fillStyle = '#212121';
ctx.fillText('Loading the model', CANVAS_SIZE / 2, CANVAS_SIZE / 2);

// Set the line color for the canvas.
ctx.strokeStyle = '#212121';

function clearCanvas() {
  ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
  for (let i = 0; i < 10; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.className = 'prediction-col';
    element.children[0].children[0].style.height = '0';
  }
}

function drawLine(fromX, fromY, toX, toY) {
  // Draws a line from (fromX, fromY) to (toX, toY).
  ctx.beginPath();
  ctx.moveTo(fromX, fromY);
  ctx.lineTo(toX, toY);
  ctx.closePath();
  ctx.stroke();
  updatePredictions();
}

async function updatePredictions() {
  // Get the predictions for the canvas data.
  let imgData = imageDataToGrayscale(ctx.getImageData(0, 0, CANVAS_SIZE, CANVAS_SIZE));
  let arr = reduceImage(imgData);
  arr = centralize(arr);

  let predictions = predict(flatten(arr));
  let maxPrediction = Math.max(...predictions);

  for (let i = 0; i < predictions.length; i++) {
    const element = document.getElementById(`prediction-${i}`);
    element.children[0].children[0].style.height = `${predictions[i] * 100}%`;
    element.className =
      predictions[i] === maxPrediction
        ? 'prediction-col top-prediction'
        : 'prediction-col';
  }
}

//////////////////////////////////////////////////////////////////

function canvasMouseDown(event) {
  isMouseDown = true;
  if (hasIntroText) {
    clearCanvas();
    hasIntroText = false;
  }
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;
  lastX = x + 0.001;
  lastY = y + 0.001;
  canvasMouseMove(event);
}

function canvasMouseMove(event) {
  const x = event.offsetX / CANVAS_SCALE;
  const y = event.offsetY / CANVAS_SCALE;
  if (isMouseDown) {
    drawLine(lastX, lastY, x, y);
  }
  lastX = x;
  lastY = y;
}

function bodyMouseUp() {
  isMouseDown = false;
}

function bodyMouseOut(event) {
  if (!event.relatedTarget || event.relatedTarget.nodeName === 'HTML') {
    isMouseDown = false;
  }
}

canvas.addEventListener('mousedown', canvasMouseDown);
canvas.addEventListener('mousemove', canvasMouseMove);
document.body.addEventListener('mouseup', bodyMouseUp);
document.body.addEventListener('mouseout', bodyMouseOut);
clearButton.addEventListener('mousedown', clearCanvas);

ctx.clearRect(0, 0, CANVAS_SIZE, CANVAS_SIZE);
ctx.fillText('Draw here!', CANVAS_SIZE / 2, CANVAS_SIZE / 2);