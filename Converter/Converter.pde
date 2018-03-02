
PImage product;
void setup() {
  size(200, 200);
  convertTraining("../training.png", "../training.txt");
  product = convertImage("../out.txt");
  surface.setResizable(true);
  surface.setSize(product.width,product.height);
}

void draw() {
  image(product, 0, 0);
}

void convertTraining(String filename, String savename) {
  PImage convert = loadImage(filename);
  String converted[] = new String[convert.width*convert.height+1];
  converted[0] = str(convert.width) + ' ' + str(convert.height);
  convert.loadPixels();
  for (int j = 0; j < convert.height; j++) for (int i = 0; i < convert.width; i++) {
    converted[1+j*convert.width + i] = str(i) + ' ' + str(j) + ' ' + str(red(convert.pixels[j*convert.width + i])/255) + ' ' + str(green(convert.pixels[j*convert.width + i])/255) + ' ' + str(blue(convert.pixels[j*convert.width + i])/255);
  }
  saveStrings(savename, converted);
}

PImage convertImage(String textfilename) {
  String[] out = loadStrings(textfilename);
  String[] scoords = out[0].split(" ", 2);
  PImage output = new PImage(int(scoords[0]), int(scoords[1]));
  output.loadPixels();
  for (int i = 1; i < out.length; i++) {
    String[] split = out[i].split(" ", 0);
    if (split.length > 4) {
      int x = int(split[0]), y = int(split[1]);
      float r = float(split[2]) * 255, g = float(split[3]) * 255, b = float(split[4]) * 255;
      output.pixels[y * output.width + x] = color(r, g, b);
    } else {
      println("Error in " + textfilename + ": line number " + i);
      exit();
    }
  }
  output.updatePixels();
  return output;
}