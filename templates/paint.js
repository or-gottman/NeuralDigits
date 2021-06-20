function _(selector){
  return document.querySelector(selector);
}
function setup(){
  let canvas = createCanvas(650, 650);
  canvas.parent("canvas");
  background(0);
}
function mouseDragged(){
  let size = 70;
  let color = "white";
  fill(color);
  stroke(color);

  ellipse(mouseX, mouseY, size, size);

}
_("#reset-canvas").addEventListener("click", function(){
  background(0);
});
_("#save-canvas").addEventListener("click",function(){
  // saveCanvas(canvas, "sketch", "png");
  makeScreenshot()
});

function makeScreenshot(){
  var canvas = _('#defaultCanvas0');
  var data = canvas.toDataURL('image/png').replace(/data:image\/png;base64,/, '');
   
  // make names  eg "img_1.png", "img_2.png"......etc"
  var iname = 'img.png'; 
   
  // _('#defaultCanvas0').remove();
  //post to php
  $.post('/recognize',{data: data, iname }, function(response){alert(response);});

  //restart sketch
  setup();
}