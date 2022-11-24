import {api} from "../api"

export async function uploadImage(base64, metadata){
    return api.post("upload", {image_to_base64: base64, metadata: metadata}).then(
        response => {return response}
    )
}

export async function predictImage(uuid, metadata){
    return api.get(`images/${uuid}`, {params: metadata}).then(
        response => {return response}
    )
}