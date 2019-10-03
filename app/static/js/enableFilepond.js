FilePond.setOptions({
    instantUpload: true,
    allowMultiple: false,
    allowReplace: false,
    allowImagePreview: true,
    server: {
      process: 'https://us-central1-fansipan-website-290191.cloudfunctions.net/classifier',
      fetch: null,
      revert: null,
      restore: null,
      load: null
    }
  });
  // FilePond.registerPlugin(
  //   FilePondPluginImagePreview
  // );
const pond = FilePond.create(document.querySelector('input[type="file"]'));
var replaced = false;
pond.on("processfile", (error, file) => {
  let prediction = document.querySelector(`#prediction`);
  let outputTable = document.querySelector(`#output`);
  let classCount = 2;
  prediction.innerHTML = 'Prediction: '
  for (i = 0; i < classCount; i++) {
    outputTable.rows[i + 1].cells[1].innerHTML = '';
    outputTable.rows[i + 1].classList.remove('is-selected');
  }
  if (error === null) {
    let data = JSON.parse(file.serverId);
    console.log(data);


    prediction.innerHTML = 'Prediction: ' + data[1].toString();
    max = data[2][0];
    max_index = 0;
    for (i = 0; i < data[2].length; i++) {
      outputTable.rows[i + 1].cells[1].innerHTML = data[2][i].toString();
      outputTable.rows[i + 1].classList.remove('is-selected');
      if (data[2][i] > max) {
        max = data[2][i];
        max_index = i;
      }

    }
    outputTable.rows[max_index + 1].classList.add('is-selected');
  }
});