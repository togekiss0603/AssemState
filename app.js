const samples={
scene:{file:'scene.png',alt:'Four scattered bench parts labeled 0, 1, 2 and 3'},
rotation:{file:'scene-rot.png',alt:'Rotated numbered view of the four bench parts'},
manual:{file:'manual.png',alt:'Original manual preview page for the applaro bench'},
step0:{file:'step-0.png',alt:'First manual step with colored masks over the bench parts'},
step1:{file:'step-1.png',alt:'Second manual step showing the existing grouping and the final side part'}
};
document.querySelectorAll('[data-sample]').forEach(button=>button.addEventListener('click',()=>{
 const item=samples[button.dataset.sample];document.querySelectorAll('[data-sample]').forEach(b=>b.setAttribute('aria-pressed',String(b===button)));
 const image=document.getElementById('sample-image');image.src='assets/'+item.file;image.alt=item.alt;document.getElementById('sample-link').href=image.src;
}));
for(const [buttonId, textId, statusId] of [['copy-citation','citation','citation-status']]) {
 document.getElementById(buttonId).addEventListener('click',async()=>{
  const status=document.getElementById(statusId);
  try { await navigator.clipboard.writeText(document.getElementById(textId).textContent); status.textContent='Copied.'; }
  catch { status.textContent='Select the text above and copy it manually.'; }
 });
}
