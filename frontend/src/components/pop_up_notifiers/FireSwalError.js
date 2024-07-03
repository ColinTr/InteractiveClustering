/*
Software Name : InteractiveClustering
Version: 1.0
SPDX-FileCopyrightText: Copyright (c) 2024 Orange
SPDX-License-Identifier: MIT
This software is distributed under the MIT License,
the text of which is available at https://spdx.org/licenses/MIT.html
or see the "license.txt" file for more details.
*/

import Swal from "sweetalert2";

const FireSwalError = function(title, text=null){
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

export default FireSwalError;