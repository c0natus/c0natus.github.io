---
title: Javascript 모달(Modal)
categories: [Program Language, Javascript]
tags: [WEB, HTML, CSS, Modal]
author: gshan
img_path: /assets/img/posts/coding/javascript/2021-07-26-how-to-make-modal-window-in-javascript/
---

Django로 웹 개발을 진행하면서 검색 기능을 구현하던 중 모달 창 구현을 해야했다.   
Javascript, css, html을 전혀 모르던 상태에서 [TCP School][1]{:target="_blank"} 사이트를 통해 빠르게 학습하고 모달 창을 구현하였다.   
모달 창을 구현하면서 겪었던 이슈에 대해서 포스트 해보고자 한다.

# Ⅰ. modal 창 구현
Modal 창을 구현하려면 html, css, javascript를 전부 알고 있어야한다. 이 모든 것을 어느정도 알고난 뒤에야 비로소 코드 이해가 가능하다.

<details>
  <summary>코드</summary>
  <div markdown="1">
  ```html
  <!DOCTYPE html>

  <head>
      <meta charset="utf-8">
      <title>modal test</title>
      <style>
          .modal-wrapper {
              position: fixed;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              background: rgba(0, 0, 0, 0.5);
              display: none;
              justify-content: center;
              align-items: center;
          }
          
          .modal-body {
              background: white;
              width: 500px;
              height: 500px;
          }
      </style>
  </head>

  <body>
      <h1>modal test 입니다.</h1>
      <button id='modal-open-btn'>열기</button>
      <div class='modal-wrapper'>
          <div class='modal-body'>
              <div class='modal-head'>
                  <h3>modal 창입니다.</h3>
                  <button id='modal-close-btn'>닫기</button>
              </div>
              <div class='modal-content'>
                  <input type='text'>
                  <button id='modal-search-btn'>검색</button>
              </div>
          </div>
      </div>
      <script>
          const modalOpenBtn = document.getElementById("modal-open-btn");
          const modalCloseBtn = document.getElementById("modal-close-btn");
          const modalWrapper = document.querySelector(".modal-wrapper");

          console.log(modalWrapper.style);

          modalOpenBtn.addEventListener("click", () => {
              modalWrapper.style.display = "flex";
          });

          modalCloseBtn.addEventListener("click", () => {
              modalWrapper.style.display = "none";
          });
      </script>
  </body>
  ```
  </div>
</details>

코드를 간단하게 설명하자면 div class wrapper를 주고, 해당 class 안에 모달 창에 표현하고 싶은 내용을 담았다. css와 javascript로 wrapper display를 none 또는 그 외의 값을 주면서 모달 기능을 구현한다.

![](1.gif)

# Ⅱ. 두번 클릭 시 이벤트 발생
어찌보면 당연한 건데, javascript에 익숙하지 않아 2시간은 씨름했던 것 같다.   
검색 결과를 table로 보여주고, 결과 데이터를 수정 및 삭제할 수 있는 버튼을 추가하는 기능을 구현하려고 했다.   
이때, 수정할 때도 모달 창을 통해 구현하려고 하였다. 그리고 수정하려는 데이터 값을 가져와서 사용자가 처음부터 일일이 데이터를 입력하지 않아도 되게 하였다.   
달라진 부분은 각 wrapper, open button, close button 마다 id를 부여하였고, javascript도 이에 대응하여 동작하도록 하였다.

<details>
  <summary>코드</summary>
  <div markdown="1">

  ```html
  <!DOCTYPE html>

  <head>
      <meta charset="utf-8">
      <title>modal test</title>
      <style>
          .modal-wrapper {
              position: fixed;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              background: rgba(0, 0, 0, 0.5);
              display: none;
              justify-content: center;
              align-items: center;
          }
          
          .modal-body {
              background: white;
              width: 500px;
              height: 500px;
          }
      </style>
  </head>

  <body>
      <h1>modal test 입니다.</h1>
      <button class='modal-open-btn' id='search_open' onclick="modal_click(this)">열기</button>
      <div class='modal-wrapper' id='search_wrp'>
          <div class='modal-body'>
              <div class='modal-head'>
                  <h3>modal 창입니다.</h3>
                  <button class='modal-close-btn' id='search_close'>닫기</button>
              </div>
              <div class='modal-content'>
                  <input type='text'>
                  <button id='modal-search-btn'>검색</button>
                  <table>
                      <thead>
                          <tr>
                              <th scope="col">이름</th>
                              <th scope="col">성별</th>
                              <th scope="col">나이</th>
                          </tr>
                      </thead>
                      <tbody>
                          <tr>
                              <td>홍길동</td>
                              <td>남</td>
                              <td>15</td>
                              <td>
                                  <button class='modal-open-btn' id='open1' onclick="modal_click(this)">수정</button>
                                  <button class='delete-btn'>삭제</button>
                                  <div class="modal-wrapper" id="wrp1">
                                      <div class='modal-body'>
                                          <div class='modal-head'>
                                              <h3>수정하는 modal 창입니다.</h3>
                                          </div>
                                          <div class='modal-content'>
                                              이름: <input type="text" value="홍길동"> 성별: <input type="text" value="남"> 나이: <input type="text" value="15">
                                              <button class='modify-modal-submit-btn' id='submot1' type='submit'>전송</button>
                                              <button class='modify-modal-close-btn' id='close1'>닫기</button>
                                          </div>
                                      </div>
                                  </div>
                              </td>
                          </tr>
                          <tr>
                              <td>홍길동</td>
                              <td>남</td>
                              <td>15</td>
                              <td>버튼 생략...</td>
                          </tr>
                          <tr>
                              <td>홍길동</td>
                              <td>남</td>
                              <td>15</td>
                              <td>버튼 생략...</td>
                          </tr>
                      </tbody>
                  </table>
              </div>
          </div>
      </div>

      <script>
          function modal_click(self) {
              if (self.id === 'search_open') {

                  var modalOpenBtn = document.getElementById(self.id);
                  var modalCloseBtn = document.getElementById('search_close');
                  var modalWrapper = document.querySelector('#search_wrp');

              } else {
                  var modalOpenBtn = document.getElementById(self.id);
                  var modalCloseBtn = document.getElementById('close' + self.id.slice(4));
                  wrp_id = '#wrp' + self.id.slice(4);
                  var modalWrapper = document.querySelector(wrp_id);
              }

              modalOpenBtn.addEventListener("click", () => {
                  modalWrapper.style.display = "flex";
              });

              modalCloseBtn.addEventListener("click", () => {
                  modalWrapper.style.display = "";
              });
          }
      </script>

  </body>
  ```

  </div>
</details>

모든 기능이 어느정도 구현이 완료되었을 때 문제가 생겼다. 바로 클릭을 두번해야 모달 창이 생성 된다는 것이다. 이벤트가 실행되기만 하면 한번의 클릭으로 모달 창이 생성되면서 원하는 방향으로 동작하였다.   
이것은 처음 javascript를 사용할 때 많이 하는 실수라고 한다. [해당 페이지][2]{:target="_blank"}를 참고하여 문제점을 해결하였다.

![](2.gif)

위의 사진을 보면 처음에 열기 버튼을 2번 클릭해야 이벤트가 동작한다.   
이는 onclick='modal_click()'을 통해 function을 호출하면서 addEventListener("click")를 통해 또, 이벤트 클릭을 기다리기 때문이다.   
따라서 addEventListener를 없애주면 문제가 해결된다.

<details>
  <summary>코드</summary>
  <div markdown="1">
  ```js
  modalOpenBtn.addEventListener("click", () => {
    modalWrapper.style.display = "flex";
  });

  <!-- 위의 코드를 아래 코드로 변경하면 된다. -->

  if (modalWrapper.style.display == "") {
    modalWrapper.style.display = "flex";
  }
  ```
  </div>
</details>

  전체 코드는 다음과 같다.
  
<details>
  <summary>코드</summary>
  <div markdown="1">
  ```html
  <!DOCTYPE html>

  <head>
      <meta charset="utf-8">
      <title>modal test</title>
      <style>
          .modal-wrapper {
              position: fixed;
              top: 0;
              left: 0;
              width: 100%;
              height: 100%;
              background: rgba(0, 0, 0, 0.5);
              display: none;
              justify-content: center;
              align-items: center;
          }
          
          .modal-body {
              background: white;
              width: 500px;
              height: 500px;
          }
      </style>
  </head>

  <body>
      <h1>modal test 입니다.</h1>
      <button class='modal-open-btn' id='search_open' onclick="modal_click(this)">열기</button>
      <div class='modal-wrapper' id='search_wrp'>
          <div class='modal-body'>
              <div class='modal-head'>
                  <h3>modal 창입니다.</h3>
                  <button class='modal-close-btn' id='search_close'>닫기</button>
              </div>
              <div class='modal-content'>
                  <input type='text'>
                  <button id='modal-search-btn'>검색</button>
                  <table>
                      <thead>
                          <tr>
                              <th scope="col">이름</th>
                              <th scope="col">성별</th>
                              <th scope="col">나이</th>
                          </tr>
                      </thead>
                      <tbody>
                          <tr>
                              <td>홍길동</td>
                              <td>남</td>
                              <td>15</td>
                              <td>
                                  <button class='modal-open-btn' id='open1' onclick="modal_click(this)">수정</button>
                                  <button class='delete-btn'>삭제</button>
                                  <div class="modal-wrapper" id="wrp1">
                                      <div class='modal-body'>
                                          <div class='modal-head'>
                                              <h3>수정하는 modal 창입니다.</h3>
                                          </div>
                                          <div class='modal-content'>
                                              이름: <input type="text" value="홍길동"> 성별: <input type="text" value="남"> 나이: <input type="text" value="15">
                                              <button class='modify-modal-submit-btn' id='submot1' type='submit'>전송</button>
                                              <button class='modify-modal-close-btn' id='close1'>닫기</button>
                                          </div>
                                      </div>
                                  </div>
                              </td>
                          </tr>
                          <tr>
                              <td>홍길동</td>
                              <td>남</td>
                              <td>15</td>
                              <td>버튼 생략...</td>
                          </tr>
                          <tr>
                              <td>홍길동</td>
                              <td>남</td>
                              <td>15</td>
                              <td>버튼 생략...</td>
                          </tr>
                      </tbody>
                  </table>
              </div>
          </div>
      </div>



      <script>
          function modal_click(self) {
              if (self.id === 'search_open') {

                  var modalOpenBtn = document.getElementById(self.id);
                  var modalCloseBtn = document.getElementById('search_close');
                  var modalWrapper = document.querySelector('#search_wrp');

              } else {
                  var modalOpenBtn = document.getElementById(self.id);
                  var modalCloseBtn = document.getElementById('close' + self.id.slice(4));
                  wrp_id = '#wrp' + self.id.slice(4);
                  var modalWrapper = document.querySelector(wrp_id);
              }

              if (modalWrapper.style.display == "") {
                  modalWrapper.style.display = "flex";
              }

              modalCloseBtn.addEventListener("click", () => {
                  modalWrapper.style.display = "";
              });
          }
      </script>

  </body>
  ```
  </div>
</details>

![](3.gif)


[1]: https://tcpschool.com/
[2]: https://okky.kr/article/546551