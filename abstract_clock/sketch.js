function setup() {
  createCanvas(800, 600);
}

function draw() {
  background('#F9F9EA'); // light yellow
  textSize(32);

  let center_x = 800/2;
  let center_y = 600/2;
  let base_size = 150;

  
  s = map(second(), 0, 60, 0, TWO_PI);
  noFill();
  stroke(color('#E0FFFF')); // light blue
  strokeWeight(20);
  arc(center_x, center_y, base_size, base_size, TWO_PI-s, TWO_PI);

  m = map(minute(), 0, 60, 0, TWO_PI);
  noFill();
  stroke(color('#B0E0E6')); // medium blue
  arc(center_x, center_y, 2*base_size, 2*base_size, TWO_PI-m, TWO_PI);

  h = map(hour(), 0, 24, 0, TWO_PI)
  noFill();
  stroke(color('#87CEEB')); // dark blue
  arc(center_x, center_y, 3*base_size, 3*base_size, TWO_PI-m, TWO_PI);
}