import Swal from "sweetalert2";

const fireSwalError = function(title, text=null){
    Swal.mixin({
        toast: true,
        position: 'top-end',
        showConfirmButton: false,
        showCloseButton: true,
        timer: 5000,
        timerProgressBar: true,
        didOpen: (toast) => {
            toast.addEventListener('mouseenter', Swal.stopTimer)
            toast.addEventListener('mouseleave', Swal.resumeTimer)
            toast.addEventListener('click', Swal.close)
        }
    }).fire({
        icon: 'error',
        title: title,
        text: text
    })
}

export default fireSwalError;