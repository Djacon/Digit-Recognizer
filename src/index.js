const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d');

let isDraw = false;

function erase() {
  isDraw = false;
  ctx.clearRect(0, 0, canvas.width, canvas.height);
}

function start(e) {
  isDraw = true;
  ctx.beginPath();
  ctx.moveTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop);
  e.preventDefault();
}

function stop(e) {
  if (isDraw) {
    isDraw = false;
    ctx.stroke();
    ctx.closePath();
  }
  e.preventDefault();
}

function draw(e) {
  if (!isDraw) return;

  ctx.lineTo(e.clientX - canvas.offsetLeft, e.clientY - canvas.offsetTop)
  ctx.lineWidth = 10;
  ctx.lineCap = 'round';
  ctx.lineJoin = 'round';
  ctx.stroke();
}

///////////////////////////////////

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

//////////////////////////////////

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

//////////////////////////////////

function recognize() {
  let t1 = new Date();

  let imgData = imageDataToGrayscale(ctx.getImageData(0, 0, 280, 280));
  let arr = reduceImage(imgData);
  let new_arr = centralize(arr);

  let ans = predict(flatten(new_arr));

  console.clear();
  let ind = 0;
  let best = 0; 
  for (let i = 0; i < 10; i++) {
    console.log(i + '. ' + (Math.round(ans[i] * 1e4) / 100) + '%');
    if (best < ans[i]) {
      best = ans[i];
      ind = i;
    }
  }

  let res = document.getElementById('result');

  if (best < 0.65) {
    ind = '¯\\_(ツ)_/¯';
    res.style = 'float:left;font-size:70px;margin-left:40px;';
  } else {
    res.style = 'float:left;font-size:200px;margin-left:80px;';
  }

  res.innerHTML = ind;

  console.log('Recognition time:', new Date() - t1 + 'ms');
}

/////////////////////////////////

erase();

canvas.addEventListener('mousemove', draw);
canvas.addEventListener('mousedown', start);
canvas.addEventListener('mouseup',   stop);
canvas.addEventListener('mouseout',  stop);
