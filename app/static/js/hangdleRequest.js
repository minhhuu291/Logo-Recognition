const renderMenu = (newmenu, idx) => {
  const { 
    image,
    name,
    price
  } = newmenu

  return  `
          <div class="card card-news">
            <img src="https://storage.cloud.google.com/fansipan-website-290191/${image}" class="card-img-top" alt="${title}">
            <div class="card-body">
              <h5 class="card-title"><a href="${url}">${name}</a></h5>
              <p class="card-text">${price}</p>
            </div>
          </div>
          `
}
//   return  `
// //           <section class="section is-white">
// //               <div class="container news-container">
// //                   <h1 class="title">What's new</h1><br>
// //                 </div>
// //                 <div class="container">
// //                   <div class="tile is-ancester">
// //                     <div class="tile is-parent is-4 is-vertical">
// //                       <div class="tile is-child"> 
// //                         <div class="card">
// //                           <div class="card-image">
// //                             <figure class="image is-square is-small">
// //                             <img src="https://storage.cloud.google.com/fansipan-website-290191/${image}" onerror="this.src='https://storage.cloud.google.com/{{bucket}}/placeholder.png'">
// //                             </figure>
// //                           </div>
// //                           <div class="card-content">
// //                             <h3 class="title is-spaced">${name}</h3>
// //                             <h3 class="subtitle">${price}$</h3>
// //                           </div>
// //                           <div class="card-footer">
// //                             <a class="card-footer-item" href="/checkout?id={{product.id}}">Purchase</a>
// //                           </div>
// //                         </div>
// //                         <br>
// //                       </div>
// //                     </div>
// //                   </div>
// //                 </div>
// //             </section>
// //             `
// }

const getNews = (menu) => {
  const data = menu
  document.getElementById('news-list').innerHTML = data.map(renderMenu).join(''); 
}

FilePond.setOptions({
    instantUpload: true,
    allowMultiple: false,
    allowReplace: false,
    allowImagePreview: true,
    server: {
      process: 'https://us-central1-fansipan-website-290191.cloudfunctions.net/classifier1',
      fetch: null,
      revert: null,
      restore: null,
      load: null
    }
  });
  FilePond.registerPlugin(
    FilePondPluginImagePreview
  );
  const pond = FilePond.create(document.querySelector('input[type="file"]'));
  pond.on('processfile', (error, file) => {
    if (error === null) {
      let data = JSON.parse(file.serverId);
      console.log(data)

      let storename = document.querySelector(`#storename`);
      let belike = document.querySelector(`#belike`);
      // let desciption = document.querySelector(`#desciption`);  
      let customer_image = document.getElementById('customer-image');
      let store_menu = document.querySelector(`#store-menu`);  
      let uploadFileIdInputNode = document.querySelector(`#image`);
      uploadFileIdInputNode.value = data;

      storename.innerHTML = data[0].toString();
      // desciption.innerHTML = data[1].toString();
      belike.innerHTML = "avg. price: " + data[4].toString() + "$";

      store_menu_link = "https://storage.cloud.google.com/fansipan-website-290191/" + data[2].toString();
      store_menu.src=store_menu_link

      customer_image_link = "https://storage.cloud.google.com/fansipan-website-290191/" + data[3].toString();
      customer_image.src=customer_image_link

      console.log(typeof data[5])
      getNews(data[5])
    }
  })